#include "gewell/console.h"
#include "bos_runner.h"

#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/bf16_primitives.h"
#include "gewell/models/gemma4/31b/model.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef GEWELL_CUDA_ARCHITECTURE
#error "GEWELL_CUDA_ARCHITECTURE must name the configured CUDA target"
#endif

namespace gewell::bos_runner {
namespace {

namespace artifact = gewell::artifact;
namespace model = gewell::gemma4_31b;
namespace primitives = gewell::bf16_primitives;

using BFloat16 = primitives::BFloat16;

constexpr std::uint32_t kBosToken = 2;
constexpr std::uint32_t kExpectedToken = 236'773;
constexpr std::size_t kCaptureCount = 114;
constexpr std::size_t kScratchAlignment = 256;

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLASLt status " + std::to_string(status));
  }
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct ScratchLayout {
  static constexpr std::size_t kH0 = 0;
  static constexpr std::size_t kH1 =
      align_up(kH0 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kH2 =
      align_up(kH1 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kQueryRaw =
      align_up(kH2 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kQueryNorm =
      align_up(kQueryRaw + model::kQueryHeadCount * model::kGlobalHeadSize *
                               sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRaw =
      align_up(kQueryNorm + model::kQueryHeadCount * model::kGlobalHeadSize *
                                sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyNorm =
      align_up(kKeyRaw + model::kLocalKvHeadCount * model::kLocalHeadSize *
                              sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueRaw =
      align_up(kKeyNorm + model::kLocalKvHeadCount * model::kLocalHeadSize *
                               sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueNorm =
      align_up(kValueRaw + model::kLocalKvHeadCount * model::kLocalHeadSize *
                                sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kContext =
      align_up(kValueNorm + model::kLocalKvHeadCount * model::kLocalHeadSize *
                                 sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGate =
      align_up(kContext + model::kQueryHeadCount * model::kGlobalHeadSize *
                              sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kUp =
      align_up(kGate + model::kMlpSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kProduct =
      align_up(kUp + model::kMlpSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kLogits =
      align_up(kProduct + model::kMlpSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kCappedLogits =
      align_up(kLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kCappedLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      kProbabilities + model::kQueryHeadCount * sizeof(BFloat16);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(ScratchLayout::kBytes == 1'341'184);

struct LayerWeightIds {
  std::size_t input_norm{};
  std::size_t q_proj{};
  std::size_t k_proj{};
  std::size_t v_proj{std::numeric_limits<std::size_t>::max()};
  std::size_t q_norm{};
  std::size_t k_norm{};
  std::size_t o_proj{};
  std::size_t post_attention_norm{};
  std::size_t pre_feedforward_norm{};
  std::size_t gate_proj{};
  std::size_t up_proj{};
  std::size_t down_proj{};
  std::size_t post_feedforward_norm{};
  std::size_t layer_scalar{};
};

constexpr std::array<LayerWeightIds, model::kLayerCount> make_weight_ids() {
  std::array<LayerWeightIds, model::kLayerCount> result{};
  for (std::size_t layer = 0; layer < result.size(); ++layer) {
    // Every preceding layer contributes 14 tensors, except each preceding
    // global layer, which deliberately has no V projection.
    const std::size_t base = 1 + 14 * layer - layer / 6;
    LayerWeightIds ids{};
    ids.input_norm = base;
    ids.q_proj = base + 1;
    ids.k_proj = base + 2;
    if (model::is_global_layer(layer)) {
      ids.q_norm = base + 3;
      ids.k_norm = base + 4;
      ids.o_proj = base + 5;
      ids.post_attention_norm = base + 6;
      ids.pre_feedforward_norm = base + 7;
      ids.gate_proj = base + 8;
      ids.up_proj = base + 9;
      ids.down_proj = base + 10;
      ids.post_feedforward_norm = base + 11;
      ids.layer_scalar = base + 12;
    } else {
      ids.v_proj = base + 3;
      ids.q_norm = base + 4;
      ids.k_norm = base + 5;
      ids.o_proj = base + 6;
      ids.post_attention_norm = base + 7;
      ids.pre_feedforward_norm = base + 8;
      ids.gate_proj = base + 9;
      ids.up_proj = base + 10;
      ids.down_proj = base + 11;
      ids.post_feedforward_norm = base + 12;
      ids.layer_scalar = base + 13;
    }
    result[layer] = ids;
  }
  return result;
}

inline constexpr auto kWeightIds = make_weight_ids();

constexpr bool tensor_is(std::size_t id, std::size_t layer,
                         model::TensorRole role) {
  return id < model::kPhysicalTensors.size() &&
         model::kPhysicalTensors[id].layer == static_cast<std::int16_t>(layer) &&
         model::kPhysicalTensors[id].role == role;
}

constexpr bool weight_ids_are_valid() {
  for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
    const LayerWeightIds& ids = kWeightIds[layer];
    if (!tensor_is(ids.input_norm, layer, model::TensorRole::input_norm) ||
        !tensor_is(ids.q_proj, layer, model::TensorRole::q_proj) ||
        !tensor_is(ids.k_proj, layer, model::TensorRole::k_proj) ||
        !tensor_is(ids.q_norm, layer, model::TensorRole::q_norm) ||
        !tensor_is(ids.k_norm, layer, model::TensorRole::k_norm) ||
        !tensor_is(ids.o_proj, layer, model::TensorRole::o_proj) ||
        !tensor_is(ids.post_attention_norm, layer,
                   model::TensorRole::post_attention_norm) ||
        !tensor_is(ids.pre_feedforward_norm, layer,
                   model::TensorRole::pre_feedforward_norm) ||
        !tensor_is(ids.gate_proj, layer, model::TensorRole::gate_proj) ||
        !tensor_is(ids.up_proj, layer, model::TensorRole::up_proj) ||
        !tensor_is(ids.down_proj, layer, model::TensorRole::down_proj) ||
        !tensor_is(ids.post_feedforward_norm, layer,
                   model::TensorRole::post_feedforward_norm) ||
        !tensor_is(ids.layer_scalar, layer,
                   model::TensorRole::layer_scalar)) {
      return false;
    }
    if (model::is_global_layer(layer)) {
      if (ids.v_proj != std::numeric_limits<std::size_t>::max()) {
        return false;
      }
    } else if (!tensor_is(ids.v_proj, layer, model::TensorRole::v_proj)) {
      return false;
    }
  }
  return model::kPhysicalTensors[model::kFinalNormPhysicalId].role ==
             model::TensorRole::final_norm &&
         model::kPhysicalTensors[model::kFinalNormPhysicalId].layer ==
             model::kNoLayer;
}

static_assert(weight_ids_are_valid());

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0) {
      fail("cudaMalloc", "zero-sized allocation");
    }
    check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] void* data() const { return pointer_; }
  [[nodiscard]] std::size_t size() const { return bytes_; }

 private:
  void* pointer_{nullptr};
  std::size_t bytes_{};
};

class LtHandle {
 public:
  LtHandle() { check_cublas(cublasLtCreate(&handle_), "cublasLtCreate"); }
  ~LtHandle() {
    if (handle_ != nullptr) {
      cublasLtDestroy(handle_);
    }
  }
  LtHandle(const LtHandle&) = delete;
  LtHandle& operator=(const LtHandle&) = delete;
  [[nodiscard]] cublasLtHandle_t get() const { return handle_; }

 private:
  cublasLtHandle_t handle_{nullptr};
};

class MatrixLayout {
 public:
  MatrixLayout(std::uint64_t rows, std::uint64_t columns,
               std::int64_t leading_dimension) {
    check_cublas(cublasLtMatrixLayoutCreate(&layout_, CUDA_R_16BF, rows,
                                            columns, leading_dimension),
                 "cublasLtMatrixLayoutCreate");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(
                     layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                     sizeof(order)),
                 "set row-major matrix layout");
  }

  ~MatrixLayout() {
    if (layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(layout_);
    }
  }
  MatrixLayout(const MatrixLayout&) = delete;
  MatrixLayout& operator=(const MatrixLayout&) = delete;
  [[nodiscard]] cublasLtMatrixLayout_t get() const { return layout_; }

 private:
  cublasLtMatrixLayout_t layout_{nullptr};
};

class LinearPlan {
 public:
  LinearPlan(std::uint32_t input_width, std::uint32_t output_width)
      : input_width_(input_width),
        output_width_(output_width),
        input_(1, input_width, input_width),
        weight_(output_width, input_width, input_width),
        output_(1, output_width, output_width) {
    check_cublas(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F),
                 "cublasLtMatmulDescCreate");
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose,
                     sizeof(no_transpose)),
                 "set BOS linear TRANSA");
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                     sizeof(transpose)),
                 "set BOS linear TRANSB");
  }

  ~LinearPlan() {
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
  }
  LinearPlan(const LinearPlan&) = delete;
  LinearPlan& operator=(const LinearPlan&) = delete;

  void run(cublasLtHandle_t handle, const BFloat16* input,
           const BFloat16* weight, BFloat16* output,
           cudaStream_t stream = nullptr) const {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasLtMatmul(handle, operation_, &alpha, input, input_.get(), weight,
                       weight_.get(), &beta, output, output_.get(), output,
                       output_.get(), nullptr, nullptr, 0, stream),
        "BOS cublasLtMatmul");
  }

 private:
  std::uint32_t input_width_{};
  std::uint32_t output_width_{};
  cublasLtMatmulDesc_t operation_{nullptr};
  MatrixLayout input_;
  MatrixLayout weight_;
  MatrixLayout output_;
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  [[nodiscard]] cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{nullptr};
};

struct CaptureSpec {
  std::string name;
  std::size_t elements{};
};

std::string layer_capture_name(std::uint32_t layer, std::string_view field) {
  char buffer[96];
  const int length = std::snprintf(buffer, sizeof(buffer),
                                   "prefill.layer.%02u.%.*s", layer,
                                   static_cast<int>(field.size()), field.data());
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
    fail("capture name", "formatted name exceeds fixed buffer");
  }
  return std::string(buffer, static_cast<std::size_t>(length));
}

void add_deep_capture_specs(std::vector<CaptureSpec>* specs,
                            std::uint32_t layer) {
  const bool global = model::is_global_layer(layer);
  const std::size_t head_size =
      global ? model::kGlobalHeadSize : model::kLocalHeadSize;
  const std::size_t kv_heads =
      global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
  const std::size_t q_width = model::kQueryHeadCount * head_size;
  const std::size_t kv_width = kv_heads * head_size;
  const auto add = [&](std::string_view field, std::size_t elements) {
    specs->push_back({layer_capture_name(layer, field), elements});
  };
  add("block_input", model::kHiddenSize);
  add("layer_scalar", 1);
  add("input_norm", model::kHiddenSize);
  add("q_raw", q_width);
  add("q_norm", q_width);
  add("q_rope", q_width);
  add("k_raw", kv_width);
  add("k_norm", kv_width);
  add("k_rope", kv_width);
  add("v_raw", kv_width);
  add("v_norm", kv_width);
  add("attention_probabilities", model::kQueryHeadCount);
  add("attention_context", q_width);
  add("attention_output", model::kHiddenSize);
  add("post_attention_norm", model::kHiddenSize);
  add("post_attention_residual", model::kHiddenSize);
  add("pre_feedforward_norm", model::kHiddenSize);
  add("mlp_gate", model::kMlpSize);
  add("mlp_up", model::kMlpSize);
  add("mlp_product", model::kMlpSize);
  add("mlp_down", model::kHiddenSize);
  add("post_feedforward_norm", model::kHiddenSize);
  add("pre_scalar_output", model::kHiddenSize);
}

std::vector<CaptureSpec> make_capture_specs() {
  std::vector<CaptureSpec> specs;
  specs.reserve(kCaptureCount);
  specs.push_back({"prefill.embedding", model::kHiddenSize});
  specs.push_back({"prefill.final_norm", model::kHiddenSize});
  specs.push_back({"prefill.logits.pre_softcap", model::kVocabSize});
  specs.push_back({"prefill.logits.post_softcap", model::kVocabSize});
  specs.push_back(
      {"prefill.rotary.sliding_attention.cos", model::kLocalHeadSize});
  specs.push_back(
      {"prefill.rotary.sliding_attention.sin", model::kLocalHeadSize});
  specs.push_back(
      {"prefill.rotary.full_attention.cos", model::kGlobalHeadSize});
  specs.push_back(
      {"prefill.rotary.full_attention.sin", model::kGlobalHeadSize});
  for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
    specs.push_back({layer_capture_name(layer, "output"), model::kHiddenSize});
  }
  add_deep_capture_specs(&specs, 0);
  add_deep_capture_specs(&specs, 5);
  std::sort(specs.begin(), specs.end(),
            [](const CaptureSpec& left, const CaptureSpec& right) {
              return left.name < right.name;
            });
  return specs;
}

struct CaptureRecord {
  std::string name;
  std::size_t elements{};
  std::size_t byte_offset{};
  bool captured{};
};

void write_exclusive(const std::filesystem::path& path, const void* data,
                     std::size_t bytes) {
  const int descriptor =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0) {
    fail("create capture file", path.string() + ": " + std::strerror(errno));
  }
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  std::size_t written = 0;
  while (written < bytes) {
    const ssize_t result = ::write(descriptor, cursor + written, bytes - written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      const std::string detail = path.string() + ": " + std::strerror(errno);
      ::close(descriptor);
      fail("write capture file", detail);
    }
    written += static_cast<std::size_t>(result);
  }
  if (::close(descriptor) != 0) {
    fail("close capture file", path.string() + ": " + std::strerror(errno));
  }
}

class CapturePlan {
 public:
  explicit CapturePlan(std::filesystem::path directory)
      : directory_(std::move(directory)) {
    std::error_code error;
    if (std::filesystem::exists(directory_, error) || error) {
      fail("capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    const std::vector<CaptureSpec> specs = make_capture_specs();
    if (specs.size() != kCaptureCount) {
      fail("capture plan", "wrong capture count");
    }
    records_.reserve(specs.size());
    std::size_t bytes = 0;
    for (const CaptureSpec& spec : specs) {
      records_.push_back({spec.name, spec.elements, bytes, false});
      bytes += spec.elements * sizeof(BFloat16);
    }
    bytes_ = bytes;
    check_cuda(cudaMallocHost(&host_arena_, bytes_),
               "cudaMallocHost BOS captures");
  }

  ~CapturePlan() {
    if (host_arena_ != nullptr) {
      cudaFreeHost(host_arena_);
    }
  }
  CapturePlan(const CapturePlan&) = delete;
  CapturePlan& operator=(const CapturePlan&) = delete;

  void copy_device(std::string_view name, const BFloat16* source,
                   cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    if (record.captured) {
      fail("capture", std::string(name) + " was produced twice");
    }
    auto* destination =
        static_cast<std::uint8_t*>(host_arena_) + record.byte_offset;
    check_cuda(cudaMemcpyAsync(destination, source,
                               record.elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "capture device-to-host copy");
    record.captured = true;
  }

  void fill_constant(std::string_view name, std::uint16_t bits) {
    CaptureRecord& record = find(name);
    if (record.captured) {
      fail("capture", std::string(name) + " was produced twice");
    }
    auto* values = reinterpret_cast<std::uint16_t*>(
        static_cast<std::uint8_t*>(host_arena_) + record.byte_offset);
    std::fill_n(values, record.elements, bits);
    record.captured = true;
  }

  void write(std::uint32_t token) {
    check_cuda(cudaDeviceSynchronize(), "synchronize BOS captures");
    for (const CaptureRecord& record : records_) {
      if (!record.captured) {
        fail("capture", record.name + " was not produced");
      }
    }
    std::error_code error;
    if (!std::filesystem::create_directory(directory_, error)) {
      fail("capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    for (const CaptureRecord& record : records_) {
      const auto* source = static_cast<const std::uint8_t*>(host_arena_) +
                           record.byte_offset;
      write_exclusive(directory_ / (record.name + ".bf16"), source,
                      record.elements * sizeof(BFloat16));
    }
    const std::array<std::uint8_t, 4> encoded_token{{
        static_cast<std::uint8_t>(token),
        static_cast<std::uint8_t>(token >> 8),
        static_cast<std::uint8_t>(token >> 16),
        static_cast<std::uint8_t>(token >> 24),
    }};
    write_exclusive(directory_ / "argmax.u32", encoded_token.data(),
                    encoded_token.size());
  }

  [[nodiscard]] std::size_t size() const { return records_.size(); }
  [[nodiscard]] const std::filesystem::path& directory() const {
    return directory_;
  }

 private:
  CaptureRecord& find(std::string_view name) {
    for (CaptureRecord& record : records_) {
      if (record.name == name) {
        return record;
      }
    }
    fail("capture", std::string(name) + " is not in the BOS inventory");
  }

  std::filesystem::path directory_;
  std::vector<CaptureRecord> records_;
  void* host_arena_{nullptr};
  std::size_t bytes_{};
};

class WeightArena {
 public:
  explicit WeightArena(const artifact::ArtifactFile& file)
      : allocation_(static_cast<std::size_t>(file.header().payload_bytes)) {
    if (allocation_.size() != artifact::kPayloadBytes) {
      fail("weight arena", "artifact payload size differs from contract");
    }
    const auto* source = file.payload_data();
    auto* destination = static_cast<std::uint8_t*>(allocation_.data());
    std::size_t copied = 0;
    while (copied < allocation_.size()) {
      const std::size_t chunk =
          std::min(artifact::kIoChunkBytes, allocation_.size() - copied);
      check_cuda(cudaMemcpy(destination + copied, source + copied, chunk,
                            cudaMemcpyHostToDevice),
                 "BF16 weight host-to-device copy");
      copied += chunk;
    }
    for (const artifact::TensorEntry& entry : file.entries()) {
      const std::uint64_t offset =
          entry.file_offset - file.header().data_offset;
      pointers_[entry.physical_id] = reinterpret_cast<const BFloat16*>(
          destination + static_cast<std::size_t>(offset));
    }
    pointers_[model::kLmHeadLogicalId] =
        pointers_[file.header().lm_head_target_id];
    if (pointers_[model::kLmHeadLogicalId] !=
        pointers_[model::kEmbeddingPhysicalId]) {
      fail("weight arena", "tied LM head does not alias the embedding");
    }
  }

  [[nodiscard]] const BFloat16* pointer(std::size_t id) const {
    if (id >= pointers_.size() || pointers_[id] == nullptr) {
      fail("weight pointer", "logical or physical id is absent");
    }
    return pointers_[id];
  }
  [[nodiscard]] std::size_t size() const { return allocation_.size(); }
  [[nodiscard]] std::uintptr_t address_mod_4096() const {
    return reinterpret_cast<std::uintptr_t>(allocation_.data()) %
           model::kStorageAlignment;
  }

 private:
  DeviceAllocation allocation_;
  std::array<const BFloat16*, model::kLogicalTensorCount> pointers_{};
};

struct LayerWeights {
  const BFloat16* input_norm{};
  const BFloat16* q_proj{};
  const BFloat16* k_proj{};
  const BFloat16* v_proj{};
  const BFloat16* q_norm{};
  const BFloat16* k_norm{};
  const BFloat16* o_proj{};
  const BFloat16* post_attention_norm{};
  const BFloat16* pre_feedforward_norm{};
  const BFloat16* gate_proj{};
  const BFloat16* up_proj{};
  const BFloat16* down_proj{};
  const BFloat16* post_feedforward_norm{};
  const BFloat16* layer_scalar{};
};

__global__ void fill_bf16_ones(BFloat16* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = __float2bfloat16_rn(1.0F);
  }
}

class BosEngine {
 public:
  explicit BosEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(ScratchLayout::kBytes),
        local_q_(model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_o_(model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(model::kHiddenSize, model::kMlpSize),
        mlp_to_hidden_(model::kMlpSize, model::kHiddenSize),
        lm_head_(model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }

  std::uint32_t run(CapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(ScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(ScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(ScratchLayout::kH2);
    BFloat16* const q_raw = at<BFloat16>(ScratchLayout::kQueryRaw);
    BFloat16* const q_norm = at<BFloat16>(ScratchLayout::kQueryNorm);
    BFloat16* const k_raw = at<BFloat16>(ScratchLayout::kKeyRaw);
    BFloat16* const k_norm = at<BFloat16>(ScratchLayout::kKeyNorm);
    BFloat16* const v_raw = at<BFloat16>(ScratchLayout::kValueRaw);
    BFloat16* const v_norm = at<BFloat16>(ScratchLayout::kValueNorm);
    BFloat16* const context = at<BFloat16>(ScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(ScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(ScratchLayout::kUp);
    BFloat16* const product = at<BFloat16>(ScratchLayout::kProduct);
    BFloat16* const logits = at<BFloat16>(ScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(ScratchLayout::kCappedLogits);
    BFloat16* const probabilities =
        at<BFloat16>(ScratchLayout::kProbabilities);
    std::uint32_t* const argmax = at<std::uint32_t>(ScratchLayout::kArgmax);

    fill_bf16_ones<<<1, 32>>>(probabilities, model::kQueryHeadCount);
    check_cuda(cudaGetLastError(), "fill singleton attention probabilities");

    if (captures != nullptr) {
      // Position zero makes both rotary families exactly the BF16 identity.
      captures->fill_constant("prefill.rotary.sliding_attention.cos", 0x3f80);
      captures->fill_constant("prefill.rotary.sliding_attention.sin", 0x0000);
      captures->fill_constant("prefill.rotary.full_attention.cos", 0x3f80);
      captures->fill_constant("prefill.rotary.full_attention.sin", 0x0000);
    }

    primitives::embedding_lookup(
        weights_.pointer(model::kEmbeddingPhysicalId), kBosToken, h0);
    capture(captures, "prefill.embedding", h0);

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const bool deep = layer == 0 || layer == 5;

      if (deep) {
        capture_layer(captures, layer, "block_input", h0);
        capture_layer(captures, layer, "layer_scalar", weight.layer_scalar);
      }

      primitives::rms_norm(h0, weight.input_norm, h1, 1,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "input_norm", h1);
      }

      const LinearPlan& q_plan = global ? global_q_ : local_q_;
      const LinearPlan& kv_plan = global ? global_kv_ : local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
      }
      if (deep) {
        capture_layer(captures, layer, "q_raw", q_raw);
        capture_layer(captures, layer, "k_raw", k_raw);
        // Global Gemma 4 deliberately feeds raw K into both normalization
        // paths; there is no global V projection or V weight.
        capture_layer(captures, layer, "v_raw", global ? k_raw : v_raw);
      }

      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           model::kQueryHeadCount, head_size);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads, head_size);
      primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm, kv_heads,
                                    head_size);
      if (deep) {
        capture_layer(captures, layer, "q_norm", q_norm);
        capture_layer(captures, layer, "q_rope", q_norm);
        capture_layer(captures, layer, "k_norm", k_norm);
        capture_layer(captures, layer, "k_rope", k_norm);
        capture_layer(captures, layer, "v_norm", v_norm);
        capture_layer(captures, layer, "attention_probabilities",
                      probabilities);
      }

      // With one causal token every head's BF16 softmax probability is 1.
      // The value product is therefore exactly the GQA head expansion.
      primitives::expand_value_heads(v_norm, context, kind);
      if (deep) {
        capture_layer(captures, layer, "attention_context", context);
      }
      const LinearPlan& o_plan = global ? global_o_ : local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2);
      if (deep) {
        capture_layer(captures, layer, "attention_output", h2);
      }

      primitives::rms_norm(h2, weight.post_attention_norm, h1, 1,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_attention_norm", h1);
      }
      primitives::residual_add(h0, h1, h2, model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_attention_residual", h2);
      }

      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, 1,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "pre_feedforward_norm", h1);
      }
      hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
      hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
      if (deep) {
        capture_layer(captures, layer, "mlp_gate", gate);
        capture_layer(captures, layer, "mlp_up", up);
      }
      primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize);
      if (deep) {
        capture_layer(captures, layer, "mlp_product", product);
      }
      mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
      if (deep) {
        capture_layer(captures, layer, "mlp_down", h0);
      }
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1, 1,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_feedforward_norm", h1);
      }
      primitives::residual_add(h2, h1, h0, model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "pre_scalar_output", h0);
      }
      primitives::trained_scalar(h0, weight.layer_scalar,
                                 model::kHiddenSize);
      capture_layer(captures, layer, "output", h0);
    }

    primitives::rms_norm(
        h0, weights_.pointer(model::kFinalNormPhysicalId), h1, 1,
        model::kHiddenSize);
    capture(captures, "prefill.final_norm", h1);
    lm_head_.run(handle_.get(), h1,
                 weights_.pointer(model::kLmHeadLogicalId), logits);
    capture(captures, "prefill.logits.pre_softcap", logits);
    primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                   model::kVocabSize, 30.0F);
    capture(captures, "prefill.logits.post_softcap", capped_logits);

    std::uint32_t result = 0;
    check_cuda(cudaMemcpy(&result, argmax, sizeof(result),
                          cudaMemcpyDeviceToHost),
               "copy BOS argmax");
    return result;
  }

 private:
  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  static void capture(CapturePlan* captures, std::string_view name,
                      const BFloat16* values) {
    if (captures != nullptr) {
      captures->copy_device(name, values);
    }
  }

  static void capture_layer(CapturePlan* captures, std::uint32_t layer,
                            std::string_view field,
                            const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(
        name, sizeof(name), "prefill.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("capture name", "formatted name exceeds fixed buffer");
    }
    captures->copy_device(std::string_view(name, static_cast<std::size_t>(length)),
                          values);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  LtHandle handle_;
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};

void validate_cuda_device() {
  constexpr std::string_view configured_arch = GEWELL_CUDA_ARCHITECTURE;
  if (configured_arch != "120a-real") {
    fail("CUDA target", "binary was not configured for 120a-real");
  }
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count < 1) {
    fail("CUDA target", "no CUDA device is available");
  }
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties");
  if (properties.major != 12 || properties.minor != 0) {
    fail("CUDA target", "device 0 is not sm_120");
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");
  console::section("GPU");
  console::field("cuda_device", properties.name);
  console::field("cuda_compute_capability",
                 std::to_string(properties.major) + "." + std::to_string(properties.minor));
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int run(const std::string& artifact_path, const std::string& capture_directory) {
  console::section("BOS inference diagnostic");
  console::field("bos_token", kBosToken);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  if (file.header().native_nvfp4 || file.header().native_mixed)
    throw std::runtime_error("BOS proof requires the BF16 artifact; use generate for native packed weights");
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before BOS load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  BosEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after BOS initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("gpu_initialization_delta_bytes", free_before - free_after);

  CudaEvent begin;
  CudaEvent end;
  const auto inference_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()), "record BOS start event");
  const std::uint32_t token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()), "record BOS end event");
  check_cuda(cudaEventSynchronize(end.get()), "synchronize BOS end event");
  float cold_inference_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&cold_inference_milliseconds, begin.get(),
                                  end.get()),
             "measure BOS inference");
  const double cold_inference_wall_seconds = seconds_since(inference_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()), "record steady BOS start event");
  const std::uint32_t steady_token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()), "record steady BOS end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady BOS end event");
  float steady_inference_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&steady_inference_milliseconds, begin.get(),
                                  end.get()),
             "measure steady BOS inference");
  const double steady_inference_wall_seconds = seconds_since(steady_started);
  if (steady_token != token) {
    fail("steady BOS replay", "argmax differs from cold run");
  }

  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_inference_milliseconds);
  console::field("cold_inference_wall_seconds", cold_inference_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_inference_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_inference_wall_seconds);
  console::field("argmax_token", token);
  console::field("expected_argmax_token", kExpectedToken);
  console::field("argmax_match", token == kExpectedToken);

  if (capture_directory != "-") {
    CapturePlan captures(capture_directory);
    const auto capture_started = std::chrono::steady_clock::now();
    const std::uint32_t capture_token = engine.run(&captures);
    const double capture_seconds = seconds_since(capture_started);
    if (capture_token != token) {
      fail("BOS capture replay", "argmax differs from timed run");
    }
    captures.write(capture_token);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return token == kExpectedToken ? 0 : 1;
}

}  // namespace gewell::bos_runner
