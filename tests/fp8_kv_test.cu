#include "gewell/bf16_primitives.h"
#include "gewell/prefill_primitives.h"
#include "gewell/mtp_attention.h"
#include "gewell/mtp_assistant.h"
#include "../src/models/gemma4/31b/sm120/cache.h"
#include "../src/models/gemma4/31b/sm120/cache_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace b = gewell::bf16_primitives;
namespace p = gewell::prefill_primitives;
namespace a = gewell::mtp_attention;
namespace m = gewell::gemma4_31b;
namespace kv = gewell::kv_cache;
using BF16 = __nv_bfloat16;
using Cache = gewell::mtp_target::CacheView;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
class Device {
 public:
  explicit Device(std::size_t bytes) : bytes(bytes) { check(cudaMalloc(&data, bytes)); }
  ~Device() { cudaFree(data); }
  template <typename T = BF16> T* get() const { return static_cast<T*>(data); }
  template <typename T> void upload(const std::vector<T>& values) {
    require(values.size() * sizeof(T) == bytes, "upload size");
    check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
  }
  std::vector<unsigned char> host() const {
    std::vector<unsigned char> result(bytes);
    check(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
    return result;
  }
  std::size_t bytes;
 private:
  void* data{};
};
BF16 bf(float x) { return __float2bfloat16_rn(x); }
float fp(BF16 x) { return __bfloat162float(x); }

// Independent exhaustive E4M3 oracle: enumerate all positive finite codes,
// then choose the nearest, resolving exact ties to an even low mantissa bit.
float decoded(unsigned code) {
  const auto e = code >> 3, fraction = code & 7;
  return e ? std::ldexp(1.0F + fraction / 8.0F, int(e) - 7)
           : std::ldexp(float(fraction), -9);
}
unsigned char encoded(float x) {
  const float magnitude = std::fabs(x);
  unsigned best = 0;
  float error = std::numeric_limits<float>::infinity();
  for (unsigned code = 0; code < 127; ++code) {
    const auto distance = std::fabs(magnitude - decoded(code));
    if (distance < error || (distance == error && !(code & 1))) {
      best = code;
      error = distance;
    }
  }
  return best | (std::signbit(x) ? 128 : 0);
}
float fp8_round(float x) {
  static const auto table = [] {
    std::array<float, 127> values{};
    for (unsigned i = 0; i < 127; ++i) values[i] = decoded(i);
    return values;
  }();
  const float magnitude = std::fabs(x);
  auto upper = std::lower_bound(table.begin(), table.end(), magnitude);
  unsigned code = upper == table.end() ? 126 : unsigned(upper - table.begin());
  if (code) {
    const float above = table[code] - magnitude, below = magnitude - table[code - 1];
    if (below < above || (below == above && (code & 1))) --code;
  }
  return std::copysign(table[code], x);
}
void quantize(const std::vector<BF16>& values, unsigned split,
              unsigned char* record, BF16* restored) {
  const unsigned n = values.size(), groups = split < n ? 2 : 1;
  for (unsigned group = 0; group < groups; ++group) {
    const auto begin = group ? split : 0U, end = group ? n : split;
    float maximum = 0;
    for (unsigned d = begin; d < end; ++d) maximum = std::max(maximum, std::fabs(fp(values[d])));
    const float scale = maximum ? maximum / 448.0F : 1.0F;
    std::memcpy(record + n + group * 4, &scale, 4);
    for (unsigned d = begin; d < end; ++d) {
      const auto code = encoded(fp(values[d]) / scale);
      record[d] = code;
      restored[d] = bf(decoded(code & 127) * (code & 128 ? -scale : scale));
    }
  }
  std::fill(record + n + groups * 4, record + kv::row_bytes(n, kv::Format::fp8, groups), 0);
}
BF16 pattern(unsigned tag, unsigned token, unsigned head, unsigned d) {
  if ((token + head) % 43 == 0) return bf(0);
  unsigned x = tag * 0x9e3779b9U + token * 0x85ebca6bU + head * 0xc2b2ae35U + d;
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
  const float magnitude = tag == 1 ? 1.0F / 2048 : std::ldexp(1.0F, int(token % 7) - 9);
  return bf((int(x & 255) - 127) * magnitude);
}

struct Fixture {
  static constexpr unsigned Capacity = 5376, Pages = 21;
  unsigned Base;
  unsigned rows, heads, d, width, capacity;
  bool global, paged;
  std::size_t stride, ref_stride;
  Device key_cache, value_cache, ref_key, ref_value, offsets, ref_offsets;
  Device query, key, value, norm, output, reference, scratch;
  Cache cache{}, ref{};
  p::CompactGlobalPagedCache page{}, ref_page{};
  std::vector<BF16> hq, hk, hv;

  Fixture(bool global, bool paged, unsigned rows, unsigned base = 1030)
      : Base(base), rows(rows), heads(global ? 4 : 16), d(global ? 512 : 256),
        width(global ? 640 : 256), capacity(global ? Capacity : 1024),
        global(global), paged(paged),
        stride(heads * 256 * kv::row_words(width, kv::Format::fp8, global ? 2 : 1) + 64),
        ref_stride(heads * 256 * width + 64),
        key_cache(paged ? (Pages * stride + 64) * 2 : heads * capacity * kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1)),
        value_cache(global ? 2 : key_cache.bytes),
        ref_key(paged ? (Pages * ref_stride + 64) * 2 : heads * capacity * width * 2),
        ref_value(global ? 2 : ref_key.bytes), offsets(Pages * 8), ref_offsets(Pages * 8),
        query(32 * rows * d * 2), key(heads * rows * d * 2), value(key.bytes),
        norm(d * 2), output(query.bytes), reference(query.bytes),
        scratch(std::max(p::tensor_attention_scratch_bytes(rows),
                        a::scratch_bytes(std::min(rows, 1280U), Capacity))) {
    std::vector<std::uint64_t> ho(Pages), hr(Pages);
    for (unsigned i = 0; i < Pages; ++i) {
      ho[i] = (Pages - 1 - i) * stride + 32;
      hr[i] = (Pages - 1 - i) * ref_stride + 32;
    }
    offsets.upload(ho); ref_offsets.upload(hr);
    cache = {key_cache.get(), global ? nullptr : value_cache.get(), capacity,
        paged ? key_cache.get() : nullptr, paged ? offsets.get<std::uint64_t>() : nullptr,
        256, Pages, stride, 32, kv::Format::fp8};
    ref = {ref_key.get(), global ? nullptr : ref_value.get(), capacity,
        paged ? ref_key.get() : nullptr, paged ? ref_offsets.get<std::uint64_t>() : nullptr,
        256, Pages, ref_stride, 32};
    page = {cache.page_pool, cache.page_offsets, 256, Pages, stride, 32, cache.format};
    ref_page = {ref.page_pool, ref.page_offsets, 256, Pages, ref_stride, 32};
    std::vector<BF16> pk(heads * Base * d), pv(pk.size());
    for (unsigned h = 0; h < heads; ++h)
      for (unsigned t = 0; t < Base; ++t)
        for (unsigned dim = 0; dim < d; ++dim) {
          pk[(h * Base + t) * d + dim] = pattern(2, t, h, dim);
          pv[(t * heads + h) * d + dim] = pattern(3, t, h, dim);
        }
    Device dk(pk.size() * 2), dv(pv.size() * 2); dk.upload(pk); dv.upload(pv);
    std::vector<unsigned char> expected(key_cache.bytes, 0xa5), expected_v(value_cache.bytes, 0xa5);
    std::vector<BF16> restored(ref_key.bytes / 2, bf(NAN)), restored_v(ref_value.bytes / 2, bf(NAN));
    key_cache.upload(expected); value_cache.upload(expected_v);
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned t = global || Base < 1024 ? 0 : Base - 1024; t < Base; ++t) {
        const auto row = std::size_t(h) * capacity + (global ? t : t % 1024);
        const auto dst = paged ? 2 * (ho[t / 256] + 32) + (h * 256 + t % 256) * kv::row_bytes(width, kv::Format::fp8, 2)
                              : row * kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1);
        const auto rd = paged ? hr[t / 256] + 32 + (h * 256 + t % 256) * width : row * width;
        std::vector<BF16> payload(width);
        if (global) {
          for (unsigned dim = 0; dim < 128; ++dim) payload[dim] = pk[(h * Base + t) * d + (dim < 64 ? dim : dim + 192)];
          std::copy_n(pv.data() + (t * heads + h) * d, d, payload.data() + 128);
          quantize(payload, 128, expected.data() + dst, restored.data() + rd);
        } else {
          std::copy_n(pk.data() + (h * Base + t) * d, d, payload.data());
          quantize(payload, d, expected.data() + dst, restored.data() + rd);
          std::copy_n(pv.data() + (t * heads + h) * d, d, payload.data());
          quantize(payload, d, expected_v.data() + dst, restored_v.data() + rd);
        }
      }
    }
    if (paged) p::write_kv_cache_chunk_global_compact_paged(dk.get(), dv.get(), page, 0, Base);
    else if (global) p::write_kv_cache_chunk_global_compact(dk.get(), dv.get(), cache.key, 0, Base, capacity, nullptr, cache.format);
    else p::write_kv_cache_chunk(dk.get(), dv.get(), cache.key, cache.value, 0, Base, capacity, m::AttentionKind::local, nullptr, cache.format);
    require(key_cache.host() == expected, "FP8 K/compact packing or guards differ from CPU oracle");
    if (!global) require(value_cache.host() == expected_v, "FP8 V packing or guards differ from CPU oracle");
    ref_key.upload(restored); ref_value.upload(restored_v);
    hq.resize(32 * rows * d); hk.resize(heads * rows * d); hv.resize(hk.size());
    for (unsigned h = 0; h < 32; ++h)
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned dim = 0; dim < d; ++dim) hq[(h * rows + r) * d + dim] = pattern(1, Base + r, h, dim);
    for (unsigned h = 0; h < heads; ++h)
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned dim = 0; dim < d; ++dim) {
          hk[(h * rows + r) * d + dim] = pattern(2, Base + r, h, dim);
          hv[(r * heads + h) * d + dim] = pattern(3, Base + r, h, dim);
        }
    query.upload(hq); key.upload(hk); value.upload(hv);
    std::vector<BF16> hn(d);
    for (unsigned dim = 0; dim < d; ++dim) hn[dim] = bf(0.75F + (dim % 31) / 64.0F);
    norm.upload(hn);
  }
  void equal(const char* label, std::size_t count = 0) {
    const auto x = output.host(), y = reference.host();
    require(!std::memcmp(x.data(), y.data(), count ? count * 2 : x.size()), label);
  }
  // Independent CPU attention oracle: scaled E4M3 Q/K, per-channel V,
  // rounded unnormalized E4M3 probabilities, causal/window masks, and online
  // accumulation across key tiles. Check multiple heads and query positions.
  void fp8_reference() {
    const auto kr = ref_key.host(), vr = ref_value.host(), actual = output.host();
    const auto* k = reinterpret_cast<const BF16*>(kr.data());
    const auto* v = reinterpret_cast<const BF16*>(vr.data());
    const auto* out = reinterpret_cast<const BF16*>(actual.data());
    for (unsigned head : {0U, 31U}) {
      const unsigned h = head / (32 / heads);
      for (unsigned r : {0U, rows / 2, rows - 1}) {
        const bool sliced = rows >= 768 && (!global || rows % 256 == 0);
        const unsigned tile_rows = 512;
        const unsigned start = sliced ? r / tile_rows * tile_rows : 0;
        const unsigned count = sliced ? std::min(tile_rows, rows - start) : rows;
        const unsigned begin = global || Base + start < 1023 ? 0 : Base + start - 1023;
        const unsigned end = Base + start + count;
        std::vector<float> q(d), numerator(d, 0);
        float qmax = 0;
        for (unsigned dim = 0; dim < d; ++dim) qmax = std::max(qmax, std::fabs(fp(hq[(head * rows + r) * d + dim])));
        const float qs = qmax > 0 ? qmax / 448 : 1;
        for (unsigned dim = 0; dim < d; ++dim) q[dim] = fp8_round(fp(hq[(head * rows + r) * d + dim]) / qs);
        float maximum = -INFINITY, denominator = 0;
        for (unsigned tile = begin; tile < end; tile += 1024) {
          const auto n = std::min(1024U, end - tile);
          std::vector<float> keys(n * d), values(n * d), ks(n), vs(d, 0), scores(n, -INFINITY);
          for (unsigned t = 0; t < n; ++t) {
            const auto pos = tile + t;
            const auto offset = paged ? (Pages - 1 - pos / 256) * ref_stride + 64 + (h * 256 + pos % 256) * width
                : (h * capacity + (global ? pos : pos % 1024)) * width;
            for (unsigned dim = 0; dim < d; ++dim) {
              const float value = pos >= Base ? fp(hv[((pos - Base) * heads + h) * d + dim])
                  : fp(global ? k[offset + 128 + dim] : v[offset + dim]);
              float key;
              if (global && !(dim < 64 || (dim >= 256 && dim < 320)))
                key = fp(bf(value * fp(bf(0.75F + (dim % 31) / 64.0F))));
              else key = pos >= Base ? fp(hk[(h * rows + pos - Base) * d + dim])
                  : fp(k[offset + (global && dim >= 256 ? dim - 192 : dim)]);
              keys[t * d + dim] = key; values[t * d + dim] = value;
              ks[t] = std::max(ks[t], std::fabs(key));
              vs[dim] = std::max(vs[dim], std::fabs(value));
            }
          }
          for (auto& x : ks) x = x > 0 ? x / 448 : 1;
          for (auto& x : vs) x = x > 0 ? x / 448 : 1;
          float tile_maximum = -INFINITY;
          for (unsigned t = 0; t < n; ++t) {
            float dot = 0;
            for (unsigned dim = 0; dim < d; ++dim) {
              dot = std::fma(q[dim], fp8_round(keys[t * d + dim] / ks[t]), dot);
              values[t * d + dim] = fp8_round(values[t * d + dim] / vs[dim]);
            }
            if (tile + t <= Base + r && (global || Base + r - (tile + t) < 1024))
              scores[t] = dot * (qs * ks[t]);
            tile_maximum = std::max(tile_maximum, scores[t]);
          }
          const float next_maximum = std::max(maximum, tile_maximum);
          const float old_scale = std::isinf(maximum) ? 0 : std::exp(maximum - next_maximum);
          for (auto& x : numerator) x *= old_scale;
          denominator *= old_scale;
          std::vector<float> partial(d, 0);
          for (unsigned t = 0; t < n; ++t) {
            const float weight = std::isinf(scores[t]) ? 0 : fp8_round(std::exp(scores[t] - next_maximum) * 256);
            denominator += weight / 256;
            for (unsigned dim = 0; dim < d; ++dim) partial[dim] = std::fma(weight, values[t * d + dim], partial[dim]);
          }
          for (unsigned dim = 0; dim < d; ++dim) numerator[dim] += partial[dim] * (vs[dim] / 256);
          maximum = next_maximum;
        }
        for (unsigned dim = 0; dim < d; ++dim) {
          const float expected = fp(bf(numerator[dim] / denominator));
          const float observed = fp(out[(r * 32 + head) * d + dim]);
          if (!std::isfinite(observed) || std::fabs(expected - observed) > 0.00004F + std::fabs(expected) * 0.012F)
            throw std::runtime_error("FP8 CPU attention oracle mismatch rows=" + std::to_string(rows) + " head=" + std::to_string(head) + " query=" + std::to_string(r) + " dim=" + std::to_string(dim) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(observed));
        }
      }
    }
  }
  void decode_batch() {
    if (rows != 1 || (global && !paged)) return;
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    Device cosine(d * 2), sine(d * 2), qr(query.bytes), kr(key.bytes);
    std::vector<BF16> cos(d, bf(0.75F)), sin(d, bf(0.25F));
    cosine.upload(cos); sine.upload(sin);
    const auto original_k = key_cache.host(), original_v = value_cache.host();
    b::apply_rope_m1(query.get(), cosine.get(), sine.get(), qr.get(), 32, kind);
    b::apply_rope_m1(key.get(), cosine.get(), sine.get(), kr.get(), heads, kind);
    if (global) {
      p::write_kv_cache_chunk_global_compact_paged(kr.get(), value.get(), page, Base, 1);
      b::causal_gqa_attention_cached_m1_fused_global_compact_paged(qr.get(), page, norm.get(), Base, scratch.get<void>(), reference.get());
    } else {
      b::write_kv_cache_m1(kr.get(), value.get(), cache.key, cache.value, Base, capacity, kind, nullptr, cache.format);
      b::causal_gqa_attention_cached_m1_fused(qr.get(), cache.key, cache.value, Base, capacity, scratch.get<void>(), reference.get(), kind, nullptr, cache.format);
    }
    const auto expected_k = key_cache.host(), expected_v = value_cache.host();
    key_cache.upload(original_k); value_cache.upload(original_v);
    b::DecodeAttentionInput input{query.get(), key.get(), value.get(), cosine.get(), sine.get(), qr.get(),
        global ? nullptr : cache.key, global ? nullptr : cache.value, page, Base, output.get(), cache.format};
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamCreate(&stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    b::decode_attention_batch({input}, norm.get(), scratch.get<void>(), scratch.bytes, kind, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    equal("FP8 batched decode differs from serial reference");
    require(key_cache.host() == expected_k && value_cache.host() == expected_v, "FP8 batched RoPE/cache write differs from serial");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
  }
  void fp8_compute() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS create");
    p::Fp8Attention fp8(rows);
    const auto invoke = [&](unsigned control, cudaStream_t stream) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle, query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(), Base, rows, scratch.get<void>(), out, stream, &fp8);
      else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle, query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, scratch.get<void>(), out, stream, c.format, &fp8);
      else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, scratch.get<void>(), out, kind, stream, c.format, &fp8);
    };
    for (unsigned control = 0; control < 2; ++control) invoke(control, nullptr);
    equal("Native FP8 attention differs between packed storage and CPU decoded storage");
    fp8_reference();
    if (global && rows == 128) {
      // Plans are warmed above. Replay the wide PV path with changed Q/K/V,
      // including changing dynamic scales and accumulation across key tiles.
      cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
      check(cudaStreamCreate(&stream));
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      invoke(0, stream);
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, 0));
      for (unsigned iteration = 0; iteration < 3; ++iteration) {
        for (auto& x : hq) x = bf(fp(x) * 0.5F - 0.03125F);
        for (auto& x : hk) x = bf(fp(x) * 1.25F);
        for (auto& x : hv) x = bf(-fp(x) * 0.75F + 0.015625F);
        query.upload(hq); key.upload(hk); value.upload(hv);
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        fp8_reference();
      }
      check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
      check(cudaStreamDestroy(stream));
      std::cout << "FP8 wide global PV changed-input graph replay CPU oracle passed\n";
    }
    cublasDestroy(handle);
  }
  void run() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS create");
    for (bool tensor : {false, true}) {
      for (unsigned control = 0; control < 2; ++control) {
        const auto& c = control ? ref : cache;
        const auto& pg = control ? ref_page : page;
        auto* out = control ? reference.get() : output.get();
        if (tensor) {
          if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle, query.get(), key.get(), value.get(), pg, norm.get(), Base, rows, scratch.get<void>(), out);
          else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle, query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, scratch.get<void>(), out, nullptr, c.format);
          else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, scratch.get<void>(), out, kind, nullptr, c.format);
        } else {
          if (paged) p::causal_gqa_attention_cached_chunk_global_compact_paged(query.get(), key.get(), value.get(), pg, norm.get(), Base, rows, out);
          else if (global) p::causal_gqa_attention_cached_chunk_global_compact(query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, out, nullptr, c.format);
          else p::causal_gqa_attention_cached_chunk(query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, out, kind, nullptr, c.format);
        }
      }
      equal(tensor ? "FP8 tensor prefill differs from decoded reference" : "FP8 scalar prefill differs from decoded reference");
    }
    cublasDestroy(handle);
    fp8_compute();
    for (unsigned control = 0; control < 2; ++control) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::image_block_gqa_attention_cached_chunk_global_compact_paged(query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(), Base, rows, Base, Base + rows, out);
      else if (global) p::image_block_gqa_attention_cached_chunk_global_compact(query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, Base, Base + rows, out, nullptr, c.format);
      else p::image_block_gqa_attention_cached_chunk(query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, Base, Base + rows, out, kind, nullptr, c.format);
    }
    equal("FP8 image attention differs from decoded reference");
    for (unsigned control = 0; control < 2; ++control)
      a::run(query.get(), key.get(), value.get(), control ? ref : cache, norm.get(), Base, rows,
          kind, control ? reference.get() : output.get(), scratch.get<void>(), scratch.bytes, nullptr);
    equal("FP8 MTP attention differs from decoded reference");
    std::vector<BF16> one(32 * d);
    for (unsigned h = 0; h < 32; ++h) std::copy_n(hq.data() + h * rows * d, d, one.data() + h * d);
    Device q(one.size() * 2); q.upload(one);
    for (unsigned control = 0; control < 2; ++control) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) b::causal_gqa_attention_cached_m1_fused_global_compact_paged(q.get(), control ? ref_page : page, norm.get(), Base - 1, scratch.get<void>(), out);
      else if (global) b::causal_gqa_attention_cached_m1_fused_global_compact(q.get(), c.key, norm.get(), Base - 1, capacity, scratch.get<void>(), out, nullptr, c.format);
      else b::causal_gqa_attention_cached_m1_fused(q.get(), c.key, c.value, Base - 1, capacity, scratch.get<void>(), out, kind, nullptr, c.format);
    }
    equal("FP8 decode differs from decoded reference", one.size());
    if (!global) {
      b::causal_gqa_attention_cached_m1_fused_local_batch(
          {{q.get(), cache.key, cache.value, Base - 1, output.get(), cache.format}},
          scratch.get<void>(), scratch.bytes);
      equal("FP8 frozen local batch differs from decoded reference", one.size());
    }
    if (global) {
      for (unsigned control = 0; control < 2; ++control)
        a::run_frozen_global(q.get(), control ? ref : cache, norm.get(), Base,
            control ? reference.get() : output.get(), scratch.get<void>(), scratch.bytes, nullptr);
      equal("FP8 frozen global attention differs from decoded reference", one.size());
    }
  }
};

// Each layer owns disjoint records; rejected rows and unwritten slots stay poisoned.
void accepted_commits(bool paged) {
  namespace t = gewell::mtp_target;
  constexpr unsigned Base = 1021, Source = 9, Capacity = 16, Accepted = 5;
  constexpr auto LocalWords = 16 * 1024 * kv::row_words(256, kv::Format::fp8);
  constexpr auto PageWords = 4 * 256 * kv::row_words(640, kv::Format::fp8, 2);
  constexpr auto GlobalWords = 5 * PageWords;
  Device staging(t::Verifier::staging_bytes(Capacity)), local(50 * 2 * LocalWords * 2), global(10 * GlobalWords * 2), offsets(5 * 8);
  std::vector<std::uint64_t> pages(5);
  for (unsigned i = 0; i < 5; ++i) pages[i] = (4 - i) * 10 * PageWords;
  offsets.upload(pages);
  std::vector<BF16> hs(staging.bytes / 2, bf(NAN));
  std::vector<unsigned char> expected_l(local.bytes, 0xa5), expected_g(global.bytes, 0xa5);
  t::Caches caches{};
  std::size_t stage = 0;
  unsigned local_layer = 0;
  for (unsigned layer = 0; layer < 60; ++layer) {
    const bool full = m::is_global_layer(layer);
    const unsigned heads = full ? 4 : 16, d = full ? 512 : 256;
    const auto value_offset = stage + std::size_t(Capacity) * heads * d;
    auto& c = caches[layer];
    c.format = kv::Format::fp8;
    c.capacity = full ? 1280 : 1024;
    if (full) {
      c.key = global.get() + (layer / 6) * GlobalWords;
      if (paged) {
        c.page_pool = global.get(); c.page_offsets = offsets.get<std::uint64_t>();
        c.page_count = 5; c.page_tokens = 256; c.page_stride_elements = 10 * PageWords;
        c.layer_offset_elements = (layer / 6) * PageWords;
      }
    } else {
      c.key = local.get() + local_layer * 2 * LocalWords;
      c.value = c.key + LocalWords;
    }
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned row = 0; row < Source; ++row) {
        for (unsigned dim = 0; dim < d; ++dim) {
          hs[stage + (h * Source + row) * d + dim] = pattern(2 + layer, row, h, dim);
          hs[value_offset + (row * heads + h) * d + dim] = pattern(3 + layer, row, h, dim);
        }
        if (row >= Accepted) continue;
        const unsigned position = Base + row;
        if (full) {
          const auto offset = paged ? pages[position / 256] + c.layer_offset_elements + (h * 256 + position % 256) * kv::row_words(640, c.format, 2)
              : (layer / 6) * GlobalWords + (h * 1280 + position) * kv::row_words(640, c.format, 2);
          std::vector<BF16> payload(640), decoded_row(640);
          for (unsigned dim = 0; dim < 128; ++dim) payload[dim] = hs[stage + (h * Source + row) * d + (dim < 64 ? dim : dim + 192)];
          std::copy_n(hs.data() + value_offset + (row * heads + h) * d, d, payload.data() + 128);
          quantize(payload, 128, expected_g.data() + 2 * offset, decoded_row.data());
        } else {
          for (unsigned kind = 0; kind < 2; ++kind) {
            const auto offset = (local_layer * 2 + kind) * LocalWords + (h * 1024 + position % 1024) * kv::row_words(256, c.format);
            const auto src = kind ? value_offset + (row * heads + h) * d : stage + (h * Source + row) * d;
            std::vector<BF16> payload(hs.begin() + src, hs.begin() + src + d), decoded_row(d);
            quantize(payload, d, expected_l.data() + 2 * offset, decoded_row.data());
          }
        }
      }
    }
    stage += 2 * std::size_t(Capacity) * heads * d;
    if (!full) ++local_layer;
  }
  staging.upload(hs);
  for (bool batch : {false, true}) {
    check(cudaMemset(local.get<void>(), 0xa5, local.bytes));
    check(cudaMemset(global.get<void>(), 0xa5, global.bytes));
    if (batch) t::commit_staged_rows_batch({{&caches, Base, Source, Capacity, staging.get<void>(), staging.bytes, Accepted}}, nullptr);
    else t::commit_staged_rows(caches, Base, Source, Capacity, staging.get<void>(), staging.bytes, Accepted, nullptr);
    require(local.host() == expected_l, "FP8 accepted local commit or untouched suffix differs from CPU oracle");
    require(global.host() == expected_g, "FP8 accepted global commit or untouched suffix differs from CPU oracle");
  }
  std::cout << "FP8 accepted serial/batched commits paged=" << paged << " passed\n";
}

__global__ void fill_storage(unsigned* data, std::size_t words) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < words) data[i] = unsigned(i * 2654435761U);
}
__global__ void check_storage(const unsigned* data, std::size_t words, unsigned* failed) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < words && data[i] != unsigned(i * 2654435761U)) atomicOr(failed, 1U);
}
void checkpoint_copies() {
  namespace s = gewell::gemma4_31b::sm120;
  const auto config = s::compact_pool_config(1536ULL << 20, 425ULL << 20,
      64ULL << 20, kv::Format::fp8, kv::Format::fp8);
  require(config.local_ring_bytes == (425ULL << 20) && config.global_page_bytes == 6717440,
      "FP8 physical cache accounting");
  kv::CacheLedger ledger(config);
  s::PhysicalCache storage(config, ledger, 1024);
  const auto bytes = config.local_ring_bytes;
  kv::ExecutionInfo source{}, destination{};
  source.processed_tokens = 1030;
  source.local_ring = ledger.try_allocate_gpu(bytes);
  destination.local_ring = ledger.try_allocate_gpu(bytes);
  const auto snapshot = ledger.try_allocate_gpu(bytes), cold = ledger.try_allocate_cpu(bytes);
  auto* src = static_cast<unsigned*>(storage.device_pointer(source.local_ring));
  auto* dst = static_cast<unsigned*>(storage.device_pointer(destination.local_ring));
  const auto words = bytes / 4;
  Device failed(4);
  fill_storage<<<(words + 255) / 256, 256>>>(src, words);
  for (unsigned mode = 0; mode < 3; ++mode) {
    check(cudaMemset(dst, 0, bytes)); check(cudaMemset(failed.get<void>(), 0, 4));
    if (mode == 0) storage.fork_local(source, destination, {});
    else {
      storage.copy_local_to_snapshot(source, 1024, snapshot, {});
      storage.synchronize({}, "test snapshot");
      if (mode == 2) storage.spill(snapshot, cold, bytes, "test spill");
      storage.restore_local(destination, mode == 2 ? cold : snapshot, 6, 1024);
    }
    check_storage<<<(words + 255) / 256, 256>>>(dst, words, failed.get<unsigned>());
    require(failed.host() == std::vector<unsigned char>(4, 0), "FP8 fork or checkpoint restore changed payload/scales");
  }
  std::cout << "FP8 ring fork, GPU snapshot, CPU spill/restore passed\n";
}
}  // namespace

int main() {
  try {
    for (unsigned rows : {1U, 4U, 8U})
      for (unsigned layout = 0; layout < 3; ++layout) {
        Fixture fixture(layout != 0, layout == 2, rows);
        fixture.run();
        fixture.decode_batch();
        std::cout << "FP8 KV layout=" << layout << " rows=" << rows << " passed\n";
      }
    for (unsigned rows : {7U, 127U, 128U, 257U, 768U, 769U, 1021U, 4096U})
      for (bool global : {false, true}) {
        Fixture fixture(global, global, rows);
        fixture.fp8_compute();
        std::cout << "FP8 compute CPU oracle global=" << global << " rows=" << rows << " passed\n";
      }
    for (unsigned rows : {1U, 31U, 513U})
      for (bool global : {false, true}) {
        Fixture fixture(global, global, rows, 1);
        fixture.fp8_compute();
        std::cout << "FP8 cold compute CPU oracle global=" << global << " rows=" << rows << " passed\n";
      }
    accepted_commits(false);
    accepted_commits(true);
    checkpoint_copies();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
