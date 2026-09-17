// Opt-in cached-prefill measurements. Inputs remain resident; L2 is not flushed.
// CUDA events cover each complete attention call, including host launch gaps.
#include "gewell/prefill_primitives.h"
#ifdef GEWELL_TEST_FUSED_PREFILL
#include "fused_prefill.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace p = gewell::prefill_primitives;
namespace m = gewell::gemma4_31b;
using BF16 = p::BFloat16;

bool fused_backend = false;
constexpr unsigned kPageTokens = 256;
constexpr unsigned kPageLayers = 10;
constexpr unsigned kWarmups = 3;
constexpr float kTolerance = 1.0F / 64;
constexpr std::size_t kLayerPageElements = 4 * kPageTokens * 640;
constexpr std::size_t kPageStrideElements = kPageLayers * kLayerPageElements;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class Device {
 public:
  explicit Device(std::size_t bytes) : bytes_(bytes) {
    check(cudaMalloc(&data_, bytes));
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template <class T = BF16> T* get() const { return static_cast<T*>(data_); }
  std::size_t bytes() const { return bytes_; }
  std::vector<BF16> host() const {
    std::vector<BF16> values(bytes_ / sizeof(BF16));
    check(cudaMemcpy(values.data(), data_, bytes_, cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  void* data_{};
  std::size_t bytes_{};
};

class Blas {
 public:
  Blas() {
    require(cublasCreate(&handle_) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
    if (cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      throw std::runtime_error("cublasSetPointerMode failed");
    }
    if (cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      throw std::runtime_error("cublasSetMathMode failed");
    }
  }
  ~Blas() { cublasDestroy(handle_); }
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;
  cublasHandle_t get() const { return handle_; }
 private:
  cublasHandle_t handle_{};
};

class Event {
 public:
  Event() { check(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }
 private:
  cudaEvent_t event_{};
};

__host__ __device__ BF16 pattern(unsigned tag, unsigned position,
                               unsigned head, unsigned dimension) {
  std::uint32_t value = tag * 0x9e3779b9U + position * 0x85ebca6bU +
                        head * 0xc2b2ae35U + dimension;
  value ^= value >> 16; value *= 0x7feb352dU;
  value ^= value >> 15; value *= 0x846ca68bU; value ^= value >> 16;
  const float scale = tag == 1 ? 1.0F / 1024 : 1.0F / 256;
  return __float2bfloat16_rn((static_cast<int>(value & 255) - 127) * scale);
}

__global__ void fill_inputs(BF16* query, BF16* key, BF16* value, BF16* norm,
                            unsigned base, unsigned rows, unsigned kv_heads, unsigned width) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < std::size_t(32) * rows * width) {
    const unsigned d = i % width, row = (i / width) % rows;
    const unsigned head = i / (std::size_t(width) * rows);
    query[i] = pattern(1, base + row, head, d);
  }
  if (i < std::size_t(kv_heads) * rows * width) {
    const unsigned d = i % width, row = (i / width) % rows;
    const unsigned head = i / (std::size_t(width) * rows);
    key[i] = pattern(2, base + row, head, d);
    value[(std::size_t(row) * kv_heads + head) * width + d] =
        pattern(3, base + row, head, d);
  }
  if (i < 512) norm[i] = __float2bfloat16_rn(0.75F + (i % 31) * (1.0F / 64));
}

__global__ void fill_local_history(BF16* cache, unsigned base) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 16ULL * 1024 * 512) return;
  const unsigned d = i % 256, value = (i / 256) % 2;
  const unsigned row = (i / 512) % 1024, head = i / (512ULL * 1024);
  const unsigned retained = min(base, 1024U);
  if (row >= retained) return;
  const unsigned position = base - retained + row, slot = position % 1024;
  cache[(std::size_t(value) * 16 * 1024 + std::size_t(head) * 1024 + slot) * 256 + d] =
      pattern(value ? 3 : 2, position, head, d);
}

__global__ void fill_global_history(p::CompactGlobalPagedCache cache, unsigned base) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 4ULL * base * 640) return;
  const unsigned d = i % 640, position = (i / 640) % base;
  const unsigned head = i / (640ULL * base);
  auto* row = gewell::compact_global_cache::paged_row(cache.page_pool,
      cache.page_offsets, cache.page_tokens, cache.layer_offset_elements, head, position);
  row[d] = d < 128 ? pattern(2, position, head, d < 64 ? d : d + 192)
                     : pattern(3, position, head, d - 128);
}

struct Fixture {
  unsigned base, rows, width, kv_heads, page_count;
  bool global;
  Device cache, offsets, query, key, value, norm, generic_output, tensor_output;
  p::CompactGlobalPagedCache pages{};

  Fixture(unsigned retained, unsigned count, bool compact_global)
      : base(retained), rows(count), width(compact_global ? 512 : 256),
        kv_heads(compact_global ? 4 : 16),
        page_count((retained + count + kPageTokens - 1) / kPageTokens),
        global(compact_global),
        cache((compact_global ? std::size_t(page_count) * kPageStrideElements
                              : 16ULL * 1024 * 2 * 256) * sizeof(BF16)),
        offsets(page_count * sizeof(std::uint64_t)),
        query(std::size_t(32) * count * width * sizeof(BF16)),
        key(std::size_t(kv_heads) * count * width * sizeof(BF16)), value(key.bytes()),
        norm(512 * sizeof(BF16)), generic_output(query.bytes()), tensor_output(query.bytes()) {
    // Poison unused page layers and future rows. Attention may only consume
    // retained history and the causal part of the separate current K/V inputs.
    check(cudaMemset(cache.get<void>(), 0xff, cache.bytes()));
    if (global) {
      std::vector<std::uint64_t> host_offsets(page_count);
      for (unsigned page = 0; page < page_count; ++page)
        host_offsets[page] = std::size_t(page_count - 1 - page) * kPageStrideElements;
      check(cudaMemcpy(offsets.get<void>(), host_offsets.data(), offsets.bytes(),
                       cudaMemcpyHostToDevice));
      pages = {cache.get(), offsets.get<std::uint64_t>(), kPageTokens, page_count,
               kPageStrideElements, (kPageLayers - 1) * kLayerPageElements};
      if (base) fill_global_history<<<(4ULL * base * 640 + 255) / 256, 256>>>(pages, base);
    } else {
      fill_local_history<<<(16ULL * 1024 * 512 + 255) / 256, 256>>>(cache.get(), base);
    }
    fill_inputs<<<(std::size_t(32) * rows * width + 255) / 256, 256>>>(
        query.get(), key.get(), value.get(), norm.get(), base, rows, kv_heads, width);
    check(cudaMemset(generic_output.get<void>(), 0xff, generic_output.bytes()));
    check(cudaMemset(tensor_output.get<void>(), 0xff, tensor_output.bytes()));
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  }

  void run(bool tensor, cublasHandle_t handle, void* scratch) {
#ifdef GEWELL_TEST_FUSED_PREFILL
    if (tensor && fused_backend) {
      if (global)
        gewell::experiments::fused_prefill::attention_global_compact_paged(
            query.get(), key.get(), value.get(), pages, norm.get(), base,
            rows, tensor_output.get());
      else
        gewell::experiments::fused_prefill::attention(query.get(), key.get(),
            value.get(), cache.get(), cache.get() + 16ULL * 1024 * 256,
            base, rows, 1024, tensor_output.get(), m::AttentionKind::local);
      return;
    }
#endif
    if (global) {
      if (tensor)
        p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle,
            query.get(), key.get(), value.get(), pages, norm.get(), base, rows,
            scratch, tensor_output.get());
      else
        p::causal_gqa_attention_cached_chunk_global_compact_paged(query.get(),
            key.get(), value.get(), pages, norm.get(), base, rows, generic_output.get());
    } else {
      if (tensor)
        p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(),
            key.get(), value.get(), cache.get(), cache.get() + 16ULL * 1024 * 256,
            base, rows, 1024, scratch, tensor_output.get(), m::AttentionKind::local);
      else
        p::causal_gqa_attention_cached_chunk(query.get(), key.get(),
            value.get(), cache.get(), cache.get() + 16ULL * 1024 * 256,
            base, rows, 1024, generic_output.get(), m::AttentionKind::local);
    }
  }

};

float validate(Fixture& fixture, cublasHandle_t handle, void* scratch) {
  fixture.run(false, handle, scratch);
  fixture.run(true, handle, scratch);
  check(cudaGetLastError());
  const auto generic = fixture.generic_output.host();
  const auto tensor = fixture.tensor_output.host();
  float maximum_error = 0;
  for (std::size_t i = 0; i < generic.size(); ++i) {
    const float expected = __bfloat162float(generic[i]);
    const float actual = __bfloat162float(tensor[i]);
    require(std::isfinite(expected) && std::isfinite(actual),
            "nonfinite output at element " + std::to_string(i));
    maximum_error = std::max(maximum_error, std::abs(actual - expected));
  }
  require(maximum_error <= kTolerance,
          "generic/tensor maximum absolute error " + std::to_string(maximum_error) +
          " exceeds 1/64");
  return maximum_error;
}

float measure(Fixture& fixture, bool tensor, cublasHandle_t handle, void* scratch,
              const Event& begin, const Event& end) {
  check(cudaEventRecord(begin.get()));
  fixture.run(tensor, handle, scratch);
  check(cudaGetLastError());
  check(cudaEventRecord(end.get()));
  check(cudaEventSynchronize(end.get()));
  float milliseconds = 0;
  check(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()));
  require(std::isfinite(milliseconds) && milliseconds > 0, "invalid event duration");
  return milliseconds;
}

float median(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
}

void emit_samples(const char* name, const std::vector<float>& samples) {
  std::cout << '"' << name << "\":{\"raw_ms\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples[i];
  }
  std::cout << "],\"median_ms\":" << median(samples)
            << ",\"min_ms\":" << *std::min_element(samples.begin(), samples.end())
            << ",\"max_ms\":" << *std::max_element(samples.begin(), samples.end()) << '}';
}

unsigned number(const std::string& text, bool base = false) {
  require(!text.empty() && std::all_of(text.begin(), text.end(),
      [](char c) { return c >= '0' && c <= '9'; }), "expected a nonnegative integer");
  const auto value = std::stoul(text);
  require(base ? value <= 262144
               : value >= 1 && value <= p::kTensorAttentionMaximumQueryRows,
          base ? "base must be in 0..262144" : "integer must be in 1..4096");
  return static_cast<unsigned>(value);
}

std::vector<unsigned> number_list(const std::string& text, bool base = false) {
  std::vector<unsigned> result;
  std::size_t start = 0;
  for (;;) {
    const auto comma = text.find(',', start);
    result.push_back(number(text.substr(start, comma == std::string::npos ? comma : comma - start), base));
    if (comma == std::string::npos) return result;
    start = comma + 1;
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<unsigned> rows{32, 128, 256, 501, 509, 512, 1024};
    std::vector<unsigned> bases{0, 20'477};
    unsigned repetitions = 11;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
#ifdef GEWELL_TEST_FUSED_PREFILL
      if (option == "--fused") { fused_backend = true; continue; }
#endif
      if (option == "--help") {
        std::cout << "Usage: prefill_chunk_benchmark [--rows 32,128,256,501,509,512,1024]"
                     " [--bases 0,20477] [--repetitions 11]\n"
                     "Rows: 1..4096. Repetitions: 1..1024. Bases: 0..262144.\n";
        return 0;
      }
      require(option == "--rows" || option == "--bases" || option == "--repetitions",
              "unknown option " + option);
      require(i + 1 < argc, "missing value for " + option);
      if (option == "--rows") rows = number_list(argv[++i]);
      else if (option == "--bases") bases = number_list(argv[++i], true);
      else {
        repetitions = number(argv[++i]);
        require(repetitions <= 1024, "repetitions must be in 1..1024");
      }
    }
    Blas blas;
    Event begin, end;
    std::cout << std::setprecision(9);
    for (unsigned base : bases) for (unsigned count : rows) for (bool global : {false, true}) {
      Device scratch(fused_backend ? 0 : p::tensor_attention_scratch_bytes(count));
      Fixture fixture(base, count, global);
      const float maximum_error = validate(fixture, blas.get(), scratch.get<void>());
      for (unsigned warmup = 0; warmup < kWarmups; ++warmup) {
        fixture.run(warmup % 2 != 0, blas.get(), scratch.get<void>());
        fixture.run(warmup % 2 == 0, blas.get(), scratch.get<void>());
      }
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      std::vector<float> generic(repetitions), tensor(repetitions);
      for (unsigned sample = 0; sample < repetitions; ++sample) {
        for (unsigned position = 0; position < 2; ++position) {
          const bool tensor_first = sample % 2 != 0;
          const bool use_tensor = position == 0 ? tensor_first : !tensor_first;
          (use_tensor ? tensor : generic)[sample] = measure(fixture, use_tensor,
              blas.get(), scratch.get<void>(), begin, end);
        }
      }
      std::cout << "{\"fused\":" << (fused_backend ? "true" : "false")
                << ",\"benchmark\":\"cached_prefill_chunk\",\"layout\":\""
                << (global ? "compact_global_paged" : "local_separate_ring")
                << "\",\"rows\":" << count << ",\"base_position\":" << base
                << ",\"cache_capacity\":" << (global ? fixture.page_count * kPageTokens : 1024)
                << ",\"page_layers\":" << (global ? kPageLayers : 0)
                << ",\"reversed_pages\":" << (global ? "true" : "false")
                << ",\"page_stride_elements\":" << (global ? kPageStrideElements : 0)
                << ",\"layer_offset_elements\":" << fixture.pages.layer_offset_elements
                << ",\"cache_allocation_bytes\":" << fixture.cache.bytes()
                << ",\"scratch_bytes\":" << scratch.bytes()
                << ",\"timing\":\"cuda_event_attention_call_span\""
                   ",\"resident_inputs\":true,\"l2_flushed\":false"
                   ",\"measurement_order\":\"alternating_generic_first\""
                   ",\"warmups_per_path\":" << kWarmups
                << ",\"repetitions_per_path\":" << repetitions
                << ",\"validation\":{\"finite\":true,\"max_abs_error\":" << maximum_error
                << ",\"absolute_tolerance\":" << kTolerance << "},";
      emit_samples("generic", generic);
      std::cout << ',';
      emit_samples("tensor", tensor);
      std::cout << ",\"generic_over_tensor_speedup\":" << median(generic) / median(tensor)
                << "}\n" << std::flush;
    }
  } catch (const std::exception& error) {
    std::cerr << "prefill chunk benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
