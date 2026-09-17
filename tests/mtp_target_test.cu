#include "gewell/mtp_target.h"
#include "gewell/models/gemma4/31b/artifact.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
namespace m = gewell::gemma4_31b;
namespace target = gewell::mtp_target;
using BF16 = target::BFloat16;
constexpr unsigned short kUntouched = 0x3f40;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
class Device {
 public:
  explicit Device(std::size_t bytes) : bytes_(bytes) { check(cudaMalloc(&data_, bytes)); }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template <typename T> T* get(std::size_t offset = 0) const {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(data_) + offset);
  }
  std::size_t bytes() const { return bytes_; }
  std::vector<unsigned short> host() const {
    std::vector<unsigned short> result(bytes_ / 2);
    check(cudaMemcpy(result.data(), data_, bytes_, cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  void* data_{};
  std::size_t bytes_{};
};

__host__ __device__ unsigned short pattern(unsigned layer, unsigned kind,
    unsigned row, unsigned head, unsigned dimension) {
  return 0x3c00 + (layer * 79 + kind * 947 + row * 41 + head * 17 + dimension * 3) % 2048;
}
__global__ void fill_bits(unsigned short* values, std::size_t count, unsigned short bits) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) values[i] = bits;
}
void fill(Device& buffer, unsigned short value) {
  fill_bits<<<(buffer.bytes() / 2 + 255) / 256, 256>>>(
      buffer.get<unsigned short>(), buffer.bytes() / 2, value);
  check(cudaGetLastError());
}
__global__ void fill_staging(unsigned short* key, unsigned short* value,
    unsigned layer, unsigned rows, unsigned heads, unsigned width) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(rows) * heads * width) return;
  const unsigned d = i % width, h = (i / width) % heads, row = i / (heads * width);
  key[(std::size_t(h) * rows + row) * width + d] = pattern(layer, 0, row, h, d);
  value[i] = pattern(layer, 1, row, h, d);
}

struct CommitFixture {
  unsigned base, source_rows, capacity_rows, accepted, global_capacity, page_count;
  bool paged;
  Device staging, local, global, page_offsets;
  target::Caches caches{};
  std::vector<std::uint64_t> offsets;
  static constexpr std::size_t kLocalKindElements = 16 * 1024 * 256;
  static constexpr std::size_t kGlobalLayerPageElements = 4 * 256 * 640;
  static constexpr std::size_t kPageStride = 10 * kGlobalLayerPageElements + 128;

  CommitFixture(unsigned b, unsigned source, unsigned capacity, unsigned count, bool pages)
      : base(b), source_rows(source), capacity_rows(capacity), accepted(count),
        global_capacity(b + source + 11), page_count((b + count + 255) / 256), paged(pages),
        staging(target::Verifier::staging_bytes(capacity)),
        local(50 * 2 * kLocalKindElements * 2),
        global((pages ? page_count * kPageStride + 64 : 10ULL * 4 * global_capacity * 640) * 2),
        page_offsets(page_count * sizeof(std::uint64_t)), offsets(page_count) {
    fill(staging, 0x7f7f); fill(local, kUntouched); fill(global, kUntouched);
    for (unsigned p = 0; p < page_count; ++p)
      offsets[p] = (page_count - 1 - p) * kPageStride + 32;
    check(cudaMemcpy(page_offsets.get<void>(), offsets.data(), page_offsets.bytes(), cudaMemcpyHostToDevice));
    std::size_t stage_offset = 0;
    unsigned local_index = 0;
    for (unsigned layer = 0; layer < 60; ++layer) {
      const bool full = m::is_global_layer(layer);
      const unsigned heads = full ? 4 : 16, width = full ? 512 : 256;
      const std::size_t elements = heads * width;
      auto* key = staging.get<unsigned short>() + stage_offset;
      auto* value = key + std::size_t(capacity) * elements;
      fill_staging<<<(std::size_t(source) * elements + 255) / 256, 256>>>(key, value, layer, source, heads, width);
      stage_offset += 2 * std::size_t(capacity) * elements;
      auto& c = caches[layer];
      if (!full) {
        c.key = local.get<BF16>() + local_index * 2 * kLocalKindElements;
        c.value = c.key + kLocalKindElements; c.capacity = 1024; ++local_index;
      } else if (paged) {
        c.page_pool = global.get<BF16>(); c.page_offsets = page_offsets.get<std::uint64_t>();
        c.page_tokens = 256; c.page_count = page_count;
        c.page_stride_elements = kPageStride;
        c.layer_offset_elements = (layer / 6) * kGlobalLayerPageElements;
      } else {
        c.key = global.get<BF16>() + (layer / 6) * std::size_t(4) * global_capacity * 640;
        c.capacity = global_capacity;
      }
    }
    check(cudaGetLastError());
  }

  void validate() {
    const auto local_result = local.host();
    auto global_expected = std::vector<unsigned short>(global.bytes() / 2, kUntouched);
    unsigned local_index = 0;
    for (unsigned layer = 0; layer < 60; ++layer) {
      if (!m::is_global_layer(layer)) {
        for (unsigned kind = 0; kind < 2; ++kind)
          for (unsigned h = 0; h < 16; ++h)
            for (unsigned slot = 0; slot < 1024; ++slot) {
              unsigned row = (slot + 1024 - base % 1024) % 1024;
              while (row + 1024 < accepted) row += 1024;
              for (unsigned d = 0; d < 256; ++d) {
                const auto expected = row < accepted ? pattern(layer, kind, row, h, d) : kUntouched;
                const auto index = (local_index * 2 + kind) * kLocalKindElements + (h * 1024 + slot) * 256 + d;
                require(local_result[index] == expected, "local accepted prefix or ring final-writer mismatch");
              }
            }
        ++local_index;
      } else {
        for (unsigned row = 0; row < accepted; ++row)
          for (unsigned h = 0; h < 4; ++h) {
            const unsigned position = base + row;
            const std::size_t offset = paged
                ? offsets[position / 256] + caches[layer].layer_offset_elements + (h * 256 + position % 256) * 640
                : ((layer / 6) * std::size_t(4) * global_capacity + h * global_capacity + position) * 640;
            for (unsigned d = 0; d < 640; ++d) {
              const unsigned source_d = d < 64 ? d : d < 128 ? d + 192 : d - 128;
              global_expected[offset + d] = pattern(layer, d < 128 ? 0 : 1, row, h, source_d);
            }
          }
      }
    }
    require(global.host() == global_expected, "global compact prefix, paged offsets, or untouched suffix mismatch");
  }
};

struct DifferentialCommitFixture {
  static constexpr std::size_t kLocalKindElements = 16 * 1024 * 256;
  static constexpr std::size_t kGlobalPageElements = 4 * 256 * 640;

  unsigned base, source_rows, capacity_rows, accepted, global_capacity,
      page_count;
  bool paged;
  Device staging, serial_local, batch_local, serial_global, batch_global,
      page_offsets;
  target::Caches serial_caches{}, batch_caches{};
  std::vector<std::uint64_t> offsets;

  DifferentialCommitFixture(unsigned b, unsigned source, unsigned capacity,
                            unsigned count, bool pages)
      : base(b),
        source_rows(source),
        capacity_rows(capacity),
        accepted(count),
        global_capacity(b + source + 11),
        page_count((b + count + 255) / 256),
        paged(pages),
        staging(target::Verifier::staging_bytes(capacity)),
        serial_local(2 * kLocalKindElements * sizeof(BF16)),
        batch_local(serial_local.bytes()),
        serial_global((pages ? page_count * kGlobalPageElements
                             : std::size_t(4) * global_capacity * 640) *
                      sizeof(BF16)),
        batch_global(serial_global.bytes()),
        page_offsets(page_count * sizeof(std::uint64_t)),
        offsets(page_count) {
    fill(staging, 0x7f7f);
    fill(serial_local, kUntouched);
    fill(batch_local, kUntouched);
    fill(serial_global, kUntouched);
    fill(batch_global, kUntouched);
    for (unsigned page = 0; page < page_count; ++page)
      offsets[page] = (page_count - 1 - page) * kGlobalPageElements;
    check(cudaMemcpy(page_offsets.get<void>(), offsets.data(),
                     page_offsets.bytes(), cudaMemcpyHostToDevice));

    std::size_t stage_offset = 0;
    for (unsigned layer = 0; layer < 60; ++layer) {
      const bool global = m::is_global_layer(layer);
      const unsigned heads = global ? 4 : 16;
      const unsigned width = global ? 512 : 256;
      const std::size_t elements = heads * width;
      auto* key = staging.get<unsigned short>() + stage_offset;
      auto* value = key + std::size_t(capacity) * elements;
      fill_staging<<<(std::size_t(source) * elements + 255) / 256, 256>>>(
          key, value, layer, source, heads, width);
      stage_offset += 2 * std::size_t(capacity) * elements;
    }
    configure(serial_caches, serial_local, serial_global);
    configure(batch_caches, batch_local, batch_global);
    check(cudaGetLastError());
  }

  void configure(target::Caches& caches, Device& local, Device& global) {
    for (unsigned layer = 0; layer < 60; ++layer) {
      auto& cache = caches[layer];
      if (!m::is_global_layer(layer)) {
        cache.key = local.get<BF16>();
        cache.value = cache.key + kLocalKindElements;
        cache.capacity = 1024;
      } else if (paged) {
        cache.page_pool = global.get<BF16>();
        cache.page_offsets = page_offsets.get<std::uint64_t>();
        cache.page_tokens = 256;
        cache.page_count = page_count;
        cache.page_stride_elements = kGlobalPageElements;
      } else {
        cache.key = global.get<BF16>();
        cache.capacity = global_capacity;
      }
    }
  }

  target::CommitInput input(const target::Caches* caches) const {
    return {caches, base, source_rows, capacity_rows, staging.get<void>(),
            staging.bytes(), accepted};
  }

  void compare() const {
    require(serial_local.host() == batch_local.host(),
            "batched local commit differs from serial");
    require(serial_global.host() == batch_global.host(),
            "batched global commit differs from serial");
  }
};

void batched_commit_differential() {
  DifferentialCommitFixture contiguous(1021, 9, 16, 5, false);
  DifferentialCommitFixture paged(255, 9, 16, 4, true);
  auto invalid = paged.batch_caches;
  invalid[59].page_count = 0;
  bool rejected = false;
  try {
    target::commit_staged_rows_batch(
        {contiguous.input(&contiguous.batch_caches), paged.input(&invalid)},
        nullptr);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "invalid batched commit was accepted");
  const auto untouched = [](const Device& buffer) {
    const auto values = buffer.host();
    return std::all_of(values.begin(), values.end(),
                       [](auto value) { return value == kUntouched; });
  };
  require(untouched(contiguous.batch_local) &&
              untouched(contiguous.batch_global) &&
              untouched(paged.batch_local) && untouched(paged.batch_global),
          "invalid batched commit partially wrote an earlier request");

  target::commit_staged_rows(
      contiguous.serial_caches, contiguous.base, contiguous.source_rows,
      contiguous.capacity_rows, contiguous.staging.get<void>(),
      contiguous.staging.bytes(), contiguous.accepted, nullptr);
  target::commit_staged_rows(
      paged.serial_caches, paged.base, paged.source_rows,
      paged.capacity_rows, paged.staging.get<void>(), paged.staging.bytes(),
      paged.accepted, nullptr);
  target::commit_staged_rows_batch(
      {contiguous.input(&contiguous.batch_caches),
       paged.input(&paged.batch_caches)},
      nullptr);
  contiguous.compare();
  paged.compare();
  std::cout << "target batch commit requests=2 mixed_counts=1 paged_and_contiguous=1 serial_exact=1\n";
}

void commit_geometry(unsigned base, unsigned source, unsigned capacity, unsigned count, bool paged) {
  CommitFixture f(base, source, capacity, count, paged);
  if (base == 1021) {
    auto invalid = f.caches;
    invalid[59].capacity = 1;
    bool rejected = false;
    try { target::commit_staged_rows(invalid, base, source, capacity,
              f.staging.get<void>(), f.staging.bytes(), count, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid last-layer cache accepted");
    const auto unchanged = f.local.host();
    require(std::all_of(unchanged.begin(), unchanged.end(), [](auto b) { return b == kUntouched; }),
            "invalid commit partially changed earlier layer caches");
  }
  target::commit_staged_rows(f.caches, base, source, capacity,
      f.staging.get<void>(), f.staging.bytes(), count, nullptr);
  f.validate();
  std::cout << "target commit base=" << base << " source=" << source
            << " capacity=" << capacity << " accepted=" << count
            << " paged=" << paged << " exact=1 untouched_suffix=1\n";
}

void projection_representations() {
  Device placeholder(256);
  target::Weights bf16{};
  bf16.fill(placeholder.get<BF16>());
  gewell::fp8::Weights fp8{};
  gewell::nvfp4::Weights nvfp4{};
  std::size_t first_fp8 = 0, mixed_mlp = 0;
  for (std::size_t i = 0; i < m::kTextPhysicalTensorCount; ++i) {
    const auto role = m::kPhysicalTensors[i].role;
    if (role == m::TensorRole::q_proj || role == m::TensorRole::k_proj ||
        role == m::TensorRole::v_proj || role == m::TensorRole::o_proj ||
        role == m::TensorRole::gate_proj || role == m::TensorRole::up_proj ||
        role == m::TensorRole::down_proj) {
      bf16[i] = nullptr;
      fp8[i] = {placeholder.get<std::uint8_t>(), 0.125F, 0.25F};
      if (!first_fp8) first_fp8 = i;
      if (!mixed_mlp && role == m::TensorRole::up_proj) mixed_mlp = i;
    }
  }
  fp8[mixed_mlp] = {};
  nvfp4[mixed_mlp] = {placeholder.get<std::uint8_t>(),
                      placeholder.get<std::uint8_t>(), 0.125F, 0.25F};
  cublasLtHandle_t handle{};
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS initialization failed");
  try {
    // Planning never reads tensor data. Cover every local/global FP8 shape,
    // one NVFP4 MLP, and BF16 norms without allocating model-sized weights.
    target::Verifier verifier(handle, bf16, 5, 16, &nvfp4,
        gewell::nvfp4::ActivationPolicy::always, &fp8);
    const auto initial_scratch = verifier.scratch_bytes();
    verifier.prepare(1);
    verifier.prepare(3);
    require(verifier.scratch_bytes() >= initial_scratch,
            "prepared FP8 shapes lost scratch accounting");
    const auto prepared_scratch = verifier.scratch_bytes();
    cudaStream_t stream{};
    cudaGraph_t graph{};
    check(cudaStreamCreate(&stream));
    try {
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      verifier.prepare(1);
      verifier.prepare(3);
      verifier.prepare(5);
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphDestroy(graph));
      graph = nullptr;
      require(verifier.scratch_bytes() == prepared_scratch,
              "repeated FP8 preparation changed scratch allocation");
    } catch (...) {
      cudaStreamCaptureStatus status{};
      if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
          status != cudaStreamCaptureStatusNone)
        cudaStreamEndCapture(stream, &graph);
      if (graph) cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      throw;
    }
    check(cudaStreamDestroy(stream));
    const auto rejects = [&] {
      bool rejected = false;
      try {
        target::Verifier invalid(handle, bf16, 1, 16, &nvfp4,
            gewell::nvfp4::ActivationPolicy::always, &fp8);
      } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "invalid target weight representation accepted");
    };
    bf16[first_fp8] = placeholder.get<BF16>();
    rejects();  // BF16 and FP8 cannot own the same projection.
    bf16[first_fp8] = nullptr;
    fp8[mixed_mlp] = fp8[first_fp8];
    rejects();  // Neither can NVFP4 and FP8.
    fp8[mixed_mlp] = {};
    const auto valid_fp8 = fp8[first_fp8];
    fp8[first_fp8] = {};
    rejects();
    for (const auto scale : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
      fp8[first_fp8] = valid_fp8;
      fp8[first_fp8].input_scale = scale;
      rejects();
      fp8[first_fp8] = valid_fp8;
      fp8[first_fp8].weight_scale = scale;
      rejects();
    }
    fp8[first_fp8] = valid_fp8;
    bf16[m::kEmbeddingPhysicalId] = nullptr;
    fp8[m::kEmbeddingPhysicalId] = valid_fp8;
    rejects();
    std::cout << "target projections mixed_bf16_nvfp4_fp8=1 all_shapes_prepared=1 invalid_representations_rejected=1\n";
  } catch (...) { cublasLtDestroy(handle); throw; }
  cublasLtDestroy(handle);
}

// Optional real-weight contract check. The cache values need not be a real
// prompt: causality and read-only verification hold for any fixed finite KV.
void real_weight_causality(const char* path) {
  const auto artifact = gewell::artifact::ArtifactFile::Open(path);
  const auto& last = artifact.entries()[m::kTextPhysicalTensorCount - 1];
  const auto weight_bytes = last.file_offset + m::align_up(last.byte_length, 4096) - artifact.header().data_offset;
  Device weights(weight_bytes);
  constexpr std::size_t chunk = 64 * 1024 * 1024;
  for (std::size_t offset = 0; offset < weight_bytes; offset += chunk)
    check(cudaMemcpy(weights.get<unsigned char>() + offset, artifact.payload_data() + offset,
                     std::min(chunk, weight_bytes - offset), cudaMemcpyHostToDevice));
  target::Weights w{};
  for (unsigned i = 0; i < m::kTextPhysicalTensorCount; ++i)
    w[i] = weights.get<BF16>(artifact.entries()[i].file_offset - artifact.header().data_offset);
  w[m::kLmHeadLogicalId] = w[0];
  cublasLtHandle_t handle{};
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS initialization failed");
  try {
    constexpr unsigned base = 1023, rows = 3, capacity = 5;
    target::Verifier verifier(handle, w, capacity);
    Device stage(target::Verifier::staging_bytes(capacity));
    Device local(2 * CommitFixture::kLocalKindElements * 2);
    Device global(4 * (base + rows) * 640 * 2);
    Device tokens(rows * sizeof(std::uint32_t));
    fill(local, 0x3c00); fill(global, 0x3b00);
    const auto original_local = local.host(), original_global = global.host();
    target::Caches caches{};
    for (unsigned i = 0; i < 60; ++i) {
      if (m::is_global_layer(i)) {
        caches[i].key = global.get<BF16>(); caches[i].capacity = base + rows;
      } else {
        caches[i].key = local.get<BF16>();
        caches[i].value = local.get<BF16>() + CommitFixture::kLocalKindElements;
        caches[i].capacity = 1024;
      }
    }
    const std::uint32_t first[rows] = {2, 902, 2172}, second[rows] = {2, 18362, 2490};
    std::vector<BF16> logits(rows * m::kVocabSize), hidden(rows * m::kHiddenSize);
    check(cudaMemcpy(tokens.get<void>(), first, sizeof(first), cudaMemcpyHostToDevice));
    verifier.run(tokens.get<std::uint32_t>(), base, rows, caches, stage.get<void>(), stage.bytes(), nullptr);
    check(cudaMemcpy(logits.data(), verifier.logits(), logits.size() * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hidden.data(), verifier.hidden(), hidden.size() * 2, cudaMemcpyDeviceToHost));
    require(local.host() == original_local && global.host() == original_global, "verifier changed committed cache");
    check(cudaMemcpy(tokens.get<void>(), second, sizeof(second), cudaMemcpyHostToDevice));
    verifier.run(tokens.get<std::uint32_t>(), base, rows, caches, stage.get<void>(), stage.bytes(), nullptr);
    std::vector<BF16> changed_logits(logits.size()), changed_hidden(hidden.size());
    check(cudaMemcpy(changed_logits.data(), verifier.logits(), logits.size() * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(changed_hidden.data(), verifier.hidden(), hidden.size() * 2, cudaMemcpyDeviceToHost));
    require(std::memcmp(logits.data(), changed_logits.data(), m::kVocabSize * 2) == 0,
            "future token changed preceding verification logits at fixed shape");
    require(std::memcmp(hidden.data(), changed_hidden.data(), m::kHiddenSize * 2) == 0,
            "future token changed preceding target hidden at fixed shape");
    require(std::memcmp(hidden.data() + m::kHiddenSize, changed_hidden.data() + m::kHiddenSize,
                        m::kHiddenSize * 2) != 0, "changed token did not exercise target execution");
    require(local.host() == original_local && global.host() == original_global, "second verifier changed committed cache");
    std::cout << "real target rows=3 capacity=5 future_token_independence=exact cache_unchanged=1\n";

    // Unequal request lengths, positions, and staging strides exercise the
    // flattened dense rows while attention must retain each request's bounds.
    constexpr unsigned other_base = 255, other_rows = 2, other_capacity = 4;
    Device other_stage(target::Verifier::staging_bytes(other_capacity));
    Device other_local(local.bytes());
    Device other_global(4 * (other_base + other_rows) * 640 * 2);
    Device batch_tokens((rows + other_rows) * sizeof(std::uint32_t));
    fill(stage, kUntouched); fill(other_stage, kUntouched);
    fill(other_local, 0x3d00); fill(other_global, 0x3a00);
    const auto original_other_local = other_local.host();
    const auto original_other_global = other_global.host();
    auto other_caches = caches;
    for (unsigned i = 0; i < 60; ++i) {
      if (m::is_global_layer(i)) {
        other_caches[i].key = other_global.get<BF16>();
        other_caches[i].capacity = other_base + other_rows;
      } else {
        other_caches[i].key = other_local.get<BF16>();
        other_caches[i].value = other_local.get<BF16>() + CommitFixture::kLocalKindElements;
      }
    }
    const std::vector<target::BatchInput> inputs = {
        {base, rows, caches, stage.get<void>(), stage.bytes(), capacity},
        {other_base, other_rows, other_caches, other_stage.get<void>(),
         other_stage.bytes(), other_capacity}};
    std::uint32_t combined[rows + other_rows] = {2, 902, 2172, 18362, 2490};
    check(cudaMemcpy(batch_tokens.get<void>(), combined, sizeof(combined), cudaMemcpyHostToDevice));
    verifier.run_batch(batch_tokens.get<std::uint32_t>(), inputs, nullptr);
    std::vector<BF16> batch_logits((rows + other_rows) * m::kVocabSize);
    std::vector<BF16> batch_hidden((rows + other_rows) * m::kHiddenSize);
    check(cudaMemcpy(batch_logits.data(), verifier.logits(), batch_logits.size() * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(batch_hidden.data(), verifier.hidden(), batch_hidden.size() * 2, cudaMemcpyDeviceToHost));
    require(local.host() == original_local && global.host() == original_global &&
                other_local.host() == original_other_local && other_global.host() == original_other_global,
            "first batched verifier changed committed caches");
    bool rejected = false;
    try { verifier.commit(caches, base, 1, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "serial commit remained enabled after run_batch");
    // Changing both tokens and the private cache of request two must not
    // change any row of request one at the same dense matrix shape.
    combined[rows] = 902; combined[rows + 1] = 2172;
    fill(other_local, 0x3c80); fill(other_global, 0x3b80);
    const auto changed_other_local = other_local.host();
    const auto changed_other_global = other_global.host();
    check(cudaMemcpy(batch_tokens.get<void>(), combined, sizeof(combined), cudaMemcpyHostToDevice));
    verifier.run_batch(batch_tokens.get<std::uint32_t>(), inputs, nullptr);
    std::vector<BF16> second_batch_logits(batch_logits.size()), second_batch_hidden(batch_hidden.size());
    check(cudaMemcpy(second_batch_logits.data(), verifier.logits(), batch_logits.size() * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(second_batch_hidden.data(), verifier.hidden(), batch_hidden.size() * 2, cudaMemcpyDeviceToHost));
    require(std::memcmp(batch_logits.data(), second_batch_logits.data(), rows * m::kVocabSize * 2) == 0,
            "another request changed preceding batch logits");
    require(std::memcmp(batch_hidden.data(), second_batch_hidden.data(), rows * m::kHiddenSize * 2) == 0,
            "another request changed preceding batch hidden");
    require(std::memcmp(batch_hidden.data() + rows * m::kHiddenSize,
                        second_batch_hidden.data() + rows * m::kHiddenSize,
                        other_rows * m::kHiddenSize * 2) != 0,
            "changed request did not exercise its own batch rows");
    require(local.host() == original_local && global.host() == original_global &&
                other_local.host() == changed_other_local && other_global.host() == changed_other_global,
            "batched verifier changed committed caches");
    for (const auto& input : inputs) {
      const auto staged = input.staging == stage.get<void>() ? stage.host() : other_stage.host();
      std::size_t offset = 0;
      for (unsigned layer = 0; layer < 60; ++layer) {
        const std::size_t width = m::is_global_layer(layer) ? 2048 : 4096;
        for (unsigned kind = 0; kind < 2; ++kind) {
          const auto start = offset + kind * input.staging_capacity_rows * width;
          require(std::all_of(staged.begin() + start + input.rows * width,
                              staged.begin() + start + input.staging_capacity_rows * width,
                              [](auto value) { return value == kUntouched; }),
                  "batched verifier overwrote inactive staging capacity");
        }
        offset += 2 * input.staging_capacity_rows * width;
      }
    }
    std::cout << "real target batch_rows=3+2 staging_capacity=5+4 request_independence=exact cache_unchanged=1\n";
  } catch (...) { cublasLtDestroy(handle); throw; }
  cublasLtDestroy(handle);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    projection_representations();
    commit_geometry(1021, 9, 16, 5, false);
    commit_geometry(253, 9, 16, 2, true); // No allocation for the rejected next page.
    commit_geometry(255, 9, 16, 4, true); // Accepted prefix crosses a page boundary.
    commit_geometry(1019, 1100, 1280, 1057, true); // More than one full ring.
    batched_commit_differential();
    if (argc == 2) real_weight_causality(argv[1]);
    else if (argc != 1) throw std::invalid_argument("usage: mtp_target_test [V3_ARTIFACT]");
    std::cout << "MTP target tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "MTP target test: " << e.what() << '\n';
    return 1;
  }
}
