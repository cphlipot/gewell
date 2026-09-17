#include "gewell/mtp_assistant.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using gewell::mtp_assistant::BFloat16;
using gewell::mtp_assistant::FrozenCache;
using gewell::mtp_assistant::attend_frozen_prefix;
using gewell::gemma4_31b::AttentionKind;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> class Device {
 public:
  explicit Device(std::size_t size) : size_(size) {
    check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
  }
  explicit Device(const std::vector<T>& data) : Device(data.size()) {
    check(cudaMemcpy(data_, data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* get() const { return data_; }
  std::vector<T> host() const {
    std::vector<T> result(size_);
    check(cudaMemcpy(result.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  T* data_{};
  std::size_t size_;
};

float value(BFloat16 x) { return __bfloat162float(x); }
BFloat16 bf16(float x) { return __float2bfloat16_rn(x); }

float compare(const std::vector<BFloat16>& actual,
              const std::vector<BFloat16>& expected, float tolerance) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
  float error = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float delta = std::abs(value(actual[i]) - value(expected[i]));
    if (!std::isfinite(delta)) throw std::runtime_error("non-finite attention output");
    error = std::max(error, delta);
  }
  if (error > tolerance) throw std::runtime_error("attention CPU error " + std::to_string(error));
  return error;
}

std::vector<BFloat16> query(unsigned width) {
  std::vector<BFloat16> result(32 * width);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = bf16((static_cast<int>((i * 13 + 3) % 97) - 48) / 256.0F);
  return result;
}

template <class Key, class Value>
std::vector<BFloat16> reference(const std::vector<BFloat16>& q,
                                unsigned width, unsigned kv_heads,
                                unsigned begin, unsigned end,
                                Key key, Value v) {
  std::vector<BFloat16> output(q.size());
  std::vector<double> scores(end - begin);
  std::vector<double> sum(width);
  for (unsigned h = 0; h < 32; ++h) {
    const unsigned kv = h / (32 / kv_heads);
    double maximum = -std::numeric_limits<double>::infinity();
    for (unsigned t = begin; t < end; ++t) {
      double score = 0;
      for (unsigned d = 0; d < width; ++d)
        score += static_cast<double>(value(q[h * width + d])) * key(kv, t, d);
      scores[t - begin] = score;
      maximum = std::max(maximum, score);
    }
    std::fill(sum.begin(), sum.end(), 0.0);
    double denominator = 0;
    for (unsigned t = begin; t < end; ++t) {
      const double probability = std::exp(scores[t - begin] - maximum);
      denominator += probability;
      for (unsigned d = 0; d < width; ++d) sum[d] += probability * v(kv, t, d);
    }
    for (unsigned d = 0; d < width; ++d)
      output[h * width + d] = bf16(static_cast<float>(sum[d] / denominator));
  }
  return output;
}

void local(unsigned length) {
  std::vector<BFloat16> k(16 * 1024 * 256), v(k.size());
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = bf16((static_cast<int>((i * 17 + 11) % 251) - 125) / 256.0F);
    v[i] = bf16((static_cast<int>((i * 7 + 43) % 239) - 119) / 128.0F);
  }
  const auto q = query(256);
  Device<BFloat16> dk(k), dv(v), dq(q), out(q.size());
  Device<unsigned char> scratch(gewell::mtp_assistant::attention_scratch_bytes(length));
  FrozenCache cache;
  cache.local_key = dk.get(); cache.local_value = dv.get(); cache.processed_tokens = length;
  attend_frozen_prefix(dq.get(), cache, nullptr, AttentionKind::local, scratch.get(), out.get());
  const auto separate = out.host();
  const auto read = [](const auto& data, unsigned head, unsigned token, unsigned dimension) {
    return value(data[(static_cast<std::size_t>(head) * 1024 + token % 1024) * 256 + dimension]);
  };
  const auto expected = reference(q, 256, 16, length > 1024 ? length - 1024 : 0, length,
      [&](unsigned h, unsigned t, unsigned d) { return read(k, h, t, d); },
      [&](unsigned h, unsigned t, unsigned d) { return read(v, h, t, d); });
  const float error = compare(separate, expected, 0.001953125F);
  if (std::memcmp(dk.host().data(), k.data(), k.size() * sizeof(BFloat16)) ||
      std::memcmp(dv.host().data(), v.data(), v.size() * sizeof(BFloat16)))
    throw std::runtime_error("assistant wrote frozen local cache");
  std::cout << "assistant local prefix=" << length << " cpu_max_abs=" << error
            << " cache_unchanged=1\n";
}

void global(unsigned length) {
  const unsigned capacity = length + 19;
  std::vector<BFloat16> compact(static_cast<std::size_t>(4) * capacity * 640), scale(512);
  for (unsigned d = 0; d < 512; ++d) scale[d] = bf16(0.5F + (d % 29) / 32.0F);
  for (std::size_t i = 0; i < compact.size(); ++i)
    compact[i] = bf16((static_cast<int>((i * 23 + 19) % 223) - 111) / 256.0F);
  const unsigned pages = (length + 255) / 256;
  const std::size_t layer_offset = 41;
  const std::size_t page_stride = layer_offset + 4 * 256 * 640 + 17;
  std::vector<BFloat16> pool(pages * page_stride + 29, bf16(17));
  std::vector<std::uint64_t> offsets(pages);
  for (unsigned p = 0; p < pages; ++p) offsets[p] = (pages - p - 1) * page_stride + 29;
  for (unsigned t = 0; t < length; ++t)
    for (unsigned h = 0; h < 4; ++h)
      std::copy_n(compact.data() + (static_cast<std::size_t>(h) * capacity + t) * 640, 640,
          pool.data() + offsets[t / 256] + layer_offset + (h * 256 + t % 256) * 640);
  const auto q = query(512);
  Device<BFloat16> dc(compact), ds(scale), dp(pool), dq(q), out(q.size());
  Device<std::uint64_t> offsets_device(offsets);
  Device<unsigned char> scratch(gewell::mtp_assistant::attention_scratch_bytes(length));
  FrozenCache cache;
  cache.processed_tokens = length; cache.global_compact = dc.get(); cache.global_capacity = capacity;
  attend_frozen_prefix(dq.get(), cache, ds.get(), AttentionKind::global, scratch.get(), out.get());
  const auto contiguous = out.host();
  const auto row = [&](unsigned h, unsigned t) {
    return compact.data() + (static_cast<std::size_t>(h) * capacity + t) * 640;
  };
  const auto expected = reference(q, 512, 4, 0, length,
      [&](unsigned h, unsigned t, unsigned d) {
        const auto* r = row(h, t);
        if (d < 64) return value(r[d]);
        if (d >= 256 && d < 320) return value(r[d - 256 + 64]);
        return value(bf16(value(r[128 + d]) * value(scale[d])));
      }, [&](unsigned h, unsigned t, unsigned d) { return value(row(h, t)[128 + d]); });
  const float error = compare(contiguous, expected, 0.001953125F);
  cache.global_compact = nullptr; cache.global_capacity = 0;
  cache.global = {dp.get(), offsets_device.get(), 256, pages, page_stride, layer_offset};
  attend_frozen_prefix(dq.get(), cache, ds.get(), AttentionKind::global, scratch.get(), out.get());
  compare(out.host(), contiguous, 0);
  if (std::memcmp(dp.host().data(), pool.data(), pool.size() * sizeof(BFloat16)))
    throw std::runtime_error("assistant wrote frozen global cache");
  std::cout << "assistant global prefix=" << length << " cpu_max_abs=" << error
            << " contiguous_paged_exact=1 noncontiguous_pages=1 cache_unchanged=1\n";
}

}  // namespace

int main() {
  try {
    for (unsigned length : {1U, 257U, 1024U, 1031U}) local(length);
    for (unsigned length : {1U, 257U, 1031U, 4099U}) global(length);
    std::cout << "assistant frozen-prefix tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "assistant test: " << error.what() << '\n';
    return 1;
  }
}
