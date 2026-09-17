#include "gewell/weight_qdq.h"
#include "weight_qdq_detail.cuh"

#include "gewell/bf16_primitives.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gewell::weight_qdq {
namespace {

using namespace detail;

[[nodiscard]] std::optional<model::TensorRole> parse_projection(
    std::string_view text) {
  for (const model::TensorRole role :
       {model::TensorRole::q_proj, model::TensorRole::k_proj,
        model::TensorRole::v_proj, model::TensorRole::o_proj,
        model::TensorRole::gate_proj, model::TensorRole::up_proj,
        model::TensorRole::down_proj}) {
    if (model::role_name(role) == text) {
      return role;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Type> parse_type(std::string_view text) {
  if (text == "bf16") {
    return Type::bf16;
  }
  if (text == "fp8") {
    return Type::fp8;
  }
  if (text == "nvfp4") {
    return Type::nvfp4;
  }
  if (text == "nvfp4_w4a4") return Type::nvfp4_w4a4;
  if (text == "fp8_w8a8") return Type::fp8_w8a8;
  return std::nullopt;
}

[[noreturn]] void parse_fail(std::string_view source_name,
                             std::size_t line,
                             std::string_view message) {
  fail(std::string(source_name) + ":" + std::to_string(line) + ": " +
       std::string(message));
}

[[nodiscard]] artifact::Digest sha256(std::string_view source) {
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr) {
    fail("QDQ mask SHA-256 context allocation failed");
  }
  artifact::Digest digest{};
  unsigned int length = 0;
  const bool ok =
      EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, source.data(), source.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &length) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || length != digest.size()) {
    fail("QDQ mask SHA-256 failed");
  }
  return digest;
}

[[nodiscard]] TypeStats& stats_for(SelectionSummary& summary, Type type) {
  switch (type) {
    case Type::bf16:
      return summary.bf16;
    case Type::fp8:
      return summary.fp8;
    case Type::nvfp4:
      return summary.nvfp4;
    case Type::nvfp4_w4a4:
      return summary.nvfp4_w4a4;
    case Type::fp8_w8a8:
      return summary.fp8_w8a8;
  }
  fail("unknown QDQ type");
}

__global__ void accumulate_absmax_kernel(const BFloat16* values,
                                         std::uint64_t elements,
                                         DeviceReductionState* state) {
  float local_maximum = 0.0F;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const float value = __bfloat162float(values[index]);
    if (!isfinite(value)) {
      atomicExch(&state->nonfinite, 1U);
    } else {
      local_maximum = fmaxf(local_maximum, fabsf(value));
    }
  }

  for (int offset = 16; offset != 0; offset /= 2) {
    local_maximum =
        fmaxf(local_maximum,
              __shfl_down_sync(0xffffffffU, local_maximum, offset));
  }
  __shared__ float warp_maxima[kThreads / 32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_maxima[warp] = local_maximum;
  }
  __syncthreads();

  if (warp == 0) {
    local_maximum =
        lane < static_cast<int>(sizeof(warp_maxima) / sizeof(warp_maxima[0]))
            ? warp_maxima[lane]
            : 0.0F;
    for (int offset = 16; offset != 0; offset /= 2) {
      local_maximum =
          fmaxf(local_maximum,
                __shfl_down_sync(0xffffffffU, local_maximum, offset));
    }
    if (lane == 0) {
      atomicMax(&state->amax_bits, __float_as_uint(local_maximum));
    }
  }
}

__global__ void fp8_qdq_kernel(BFloat16* values, std::uint64_t elements,
                               float scale) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const float normalized = __bfloat162float(values[index]) / scale;
    values[index] = round_bf16(round_e4m3(normalized) * scale);
  }
}

__global__ void nvfp4_qdq_kernel(BFloat16* values, std::uint64_t elements,
                                 float global_scale) {
  std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  const std::uint64_t iterations = (elements + stride - 1) / stride;
  for (std::uint64_t iteration = 0; iteration < iterations;
       ++iteration, index += stride) {
    const bool valid = index < elements;
    const float value = valid ? __bfloat162float(values[index]) : 0.0F;
    float block_amax = fabsf(value);
    for (int offset = 8; offset != 0; offset /= 2) {
      block_amax =
          fmaxf(block_amax,
                __shfl_down_sync(0xffffffffU, block_amax, offset, 16));
    }
    block_amax = __shfl_sync(0xffffffffU, block_amax, 0, 16);

    const float block_scale =
        nvfp4_block_scale(block_amax, global_scale);
    if (valid) {
      const float combined_scale = block_scale * global_scale;
      const float normalized = value / combined_scale;
      values[index] =
          round_bf16(round_e2m1(normalized) * combined_scale);
    }
  }
}

[[nodiscard]] int reduction_blocks(std::uint64_t elements) {
  const std::uint64_t needed =
      (elements + static_cast<std::uint64_t>(kThreads) - 1) / kThreads;
  return static_cast<int>(
      std::max<std::uint64_t>(1, std::min<std::uint64_t>(
                                     needed, kMaximumReductionBlocks)));
}

[[nodiscard]] float device_absmax(DeviceState& state,
                                  const BFloat16* values,
                                  std::uint64_t elements) {
  check_cuda(cudaMemset(state.get(), 0, sizeof(DeviceReductionState)),
             "cudaMemset QDQ reduction state");
  accumulate_absmax_kernel<<<reduction_blocks(elements), kThreads>>>(
      values, elements, state.get());
  check_cuda(cudaGetLastError(), "launch QDQ absmax kernel");
  DeviceReductionState host{};
  check_cuda(cudaMemcpy(&host, state.get(), sizeof(host),
                        cudaMemcpyDeviceToHost),
             "copy QDQ reduction state");
  if (host.nonfinite != 0) {
    fail("selected QDQ weight contains a non-finite BF16 value");
  }
  float result = 0.0F;
  static_assert(sizeof(result) == sizeof(host.amax_bits));
  std::memcpy(&result, &host.amax_bits, sizeof(result));
  return result;
}

}  // namespace

void detail::apply_tensor(DeviceState& state, Type type, BFloat16* values,
                  std::uint64_t elements) {
  if (type == Type::bf16 || elements == 0) {
    return;
  }
  const float amax = device_absmax(state, values, elements);
  if (amax == 0.0F) {
    return;
  }
  switch (type) {
    case Type::bf16:
      return;
    case Type::nvfp4_w4a4:
    case Type::fp8_w8a8:
      fail("packed storage requires native execution, not QDQ");
    case Type::fp8: {
      const float scale = amax / kE4m3Maximum;
      fp8_qdq_kernel<<<reduction_blocks(elements), kThreads>>>(values,
                                                               elements,
                                                               scale);
      check_cuda(cudaGetLastError(), "launch static FP8 QDQ kernel");
      return;
    }
    case Type::nvfp4: {
      if (elements % 16 != 0) {
        fail("NVFP4 QDQ tensor element count is not divisible by 16");
      }
      const float global_scale =
          amax / (kE2m1Maximum * kE4m3Maximum);
      nvfp4_qdq_kernel<<<reduction_blocks(elements), kThreads>>>(
          values, elements, global_scale);
      check_cuda(cudaGetLastError(), "launch NVFP4 QDQ kernel");
      return;
    }
  }
  fail("unknown QDQ type");
}


Mask Mask::Parse(std::string_view source, std::string source_name) {
  if (source.size() > kMaximumMaskBytes) {
    fail(source_name + ": QDQ mask exceeds 1 MiB");
  }
  Mask result;
  result.has_source_ = true;
  result.source_name_ = std::move(source_name);
  result.source_sha256_ = sha256(source);

  std::istringstream input{std::string(source)};
  std::string line_text;
  std::size_t line_number = 0;
  while (std::getline(input, line_text)) {
    ++line_number;
    const std::size_t comment = line_text.find('#');
    if (comment != std::string::npos) {
      line_text.resize(comment);
    }
    std::istringstream line(line_text);
    std::string layer_text;
    std::string projection_text;
    std::string type_text;
    std::string extra;
    if (!(line >> layer_text)) {
      continue;
    }
    if (!(line >> projection_text >> type_text) || (line >> extra)) {
      parse_fail(result.source_name_, line_number,
                 "expected exactly: LAYER PROJECTION TYPE");
    }

    const std::optional<model::TensorRole> role =
        parse_projection(projection_text);
    if (!role.has_value()) {
      parse_fail(result.source_name_, line_number,
                 "unknown projection '" + projection_text + "'");
    }
    const std::optional<Type> type = parse_type(type_text);
    if (!type.has_value()) {
      parse_fail(result.source_name_, line_number,
                 "unknown QDQ type '" + type_text + "'");
    }
    const std::size_t projection = *projection_index(*role);
    if (layer_text == "*") {
      for (std::uint32_t layer_index = 0; layer_index < model::kLayerCount;
           ++layer_index) {
        if (*role == model::TensorRole::v_proj &&
            model::is_global_layer(layer_index)) {
          continue;
        }
        result.types_[layer_index][projection] = *type;
      }
      continue;
    }

    std::uint32_t layer_index = 0;
    const char* const begin = layer_text.data();
    const char* const end = begin + layer_text.size();
    const auto parsed = std::from_chars(begin, end, layer_index);
    if (layer_text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != end || layer_index >= model::kLayerCount) {
      parse_fail(result.source_name_, line_number,
                 "layer must be '*' or an integer in 0..59");
    }
    if (*role == model::TensorRole::v_proj &&
        model::is_global_layer(layer_index)) {
      parse_fail(result.source_name_, line_number,
                 "global layers do not have v_proj");
    }
    result.types_[layer_index][projection] = *type;
  }
  return result;
}

Mask Mask::Load(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    fail("open QDQ mask " + path + " failed");
  }
  const std::streampos end = input.tellg();
  if (end < 0) {
    fail("size QDQ mask " + path + " failed");
  }
  if (static_cast<std::uint64_t>(end) > kMaximumMaskBytes) {
    fail(path + ": QDQ mask exceeds 1 MiB");
  }
  std::string source(static_cast<std::size_t>(end), '\0');
  input.seekg(0);
  if (!source.empty()) {
    input.read(source.data(), static_cast<std::streamsize>(source.size()));
  }
  if (!input) {
    fail("read QDQ mask " + path + " failed");
  }
  return Parse(source, path);
}

Type Mask::type_for(const model::TensorSpec& tensor) const {
  if (tensor.layer < 0 || tensor.layer >= model::kLayerCount) {
    return Type::bf16;
  }
  return type_for(static_cast<std::uint32_t>(tensor.layer), tensor.role);
}

Type Mask::type_for(std::uint32_t layer, model::TensorRole role) const {
  if (layer >= model::kLayerCount) {
    fail("QDQ lookup layer is outside 0..59");
  }
  const std::optional<std::size_t> projection = projection_index(role);
  if (!projection.has_value()) {
    return Type::bf16;
  }
  if (role == model::TensorRole::v_proj && model::is_global_layer(layer)) {
    fail("QDQ lookup requested nonexistent global v_proj");
  }
  return types_[layer][*projection];
}

bool Mask::all_bf16() const {
  for (const auto& layer : types_) {
    for (const Type type : layer) {
      if (type != Type::bf16) {
        return false;
      }
    }
  }
  return true;
}

bool Mask::has_qdq() const {
  for (const auto& layer : types_)
    for (const auto type : layer)
      if (type == Type::fp8 || type == Type::nvfp4) return true;
  return false;
}

std::string_view type_name(Type type) {
  switch (type) {
    case Type::bf16:
      return "bf16";
    case Type::fp8:
      return "fp8";
    case Type::nvfp4:
      return "nvfp4";
    case Type::nvfp4_w4a4:
      return "nvfp4_w4a4";
    case Type::fp8_w8a8:
      return "fp8_w8a8";
  }
  return "unknown";
}

SelectionSummary summarize(const Mask& mask) {
  SelectionSummary result;
  for (const model::TensorSpec& tensor : model::kPhysicalTensors) {
    if (!projection_index(tensor.role).has_value()) {
      continue;
    }
    TypeStats& stats = stats_for(result, mask.type_for(tensor));
    ++stats.tensor_count;
    stats.element_count += tensor.shape.element_count();
    stats.source_bf16_bytes += tensor.byte_count();
  }
  return result;
}

ApplySummary apply_in_place(const Mask& mask,
                            const artifact::ArtifactFile& file,
                            void* device_payload) {
  ApplySummary result;
  result.selection = summarize(mask);
  // Native precision is stored in the artifact. A supplied mask asserts that
  // representation; changing native/BF16 selection requires offline repacking.
  for (const auto& entry : file.entries()) {
    const auto& tensor = model::kPhysicalTensors[entry.physical_id];
    const auto selected = mask.type_for(tensor);
    const auto stored = entry.storage_type == artifact::StorageType::nvfp4_w4a4
        ? Type::nvfp4_w4a4 : entry.storage_type == artifact::StorageType::fp8_w8a8
        ? Type::fp8_w8a8 : Type::bf16;
    const bool native = stored != Type::bf16;
    if ((selected == Type::nvfp4_w4a4 || selected == Type::fp8_w8a8) && selected != stored)
      fail(std::string(type_name(selected)) + " mask requires matching packed artifact storage");
    if (native && mask.has_source() && selected != stored)
      fail("mask differs from packed artifact precision; repack with the desired mask");
    if (native && !mask.has_source()) {
      auto& bf16 = result.selection.bf16;
      auto& packed = stats_for(result.selection, stored);
      --bf16.tensor_count;
      bf16.element_count -= tensor.shape.element_count();
      bf16.source_bf16_bytes -= tensor.byte_count();
      ++packed.tensor_count;
      packed.element_count += tensor.shape.element_count();
      packed.source_bf16_bytes += tensor.byte_count();
    }
  }
  if (!mask.has_qdq()) return result;
  if (device_payload == nullptr) {
    fail("QDQ device payload is null");
  }

  const auto started = std::chrono::steady_clock::now();
  {
    DeviceState state;
    auto* const base = static_cast<std::uint8_t*>(device_payload);
    for (const artifact::TensorEntry& entry : file.entries()) {
      const model::TensorSpec& tensor =
          model::kPhysicalTensors[entry.physical_id];
      const Type type = mask.type_for(tensor);
      if (type == Type::bf16 || type == Type::nvfp4_w4a4 || type == Type::fp8_w8a8) {
        continue;
      }
      if (tensor.shape.rank != 2 || tensor.layer < 0 ||
          entry.byte_length != tensor.byte_count()) {
        fail(
            "selected QDQ tensor does not match the compiled projection contract");
      }
      if (type == Type::nvfp4 && tensor.shape.dimensions[1] % 16 != 0) {
        fail("selected NVFP4 projection input width is not divisible by 16");
      }
      const std::uint64_t offset =
          entry.file_offset - file.header().data_offset;
      auto* const values = reinterpret_cast<BFloat16*>(
          base + static_cast<std::size_t>(offset));
      apply_tensor(state, type, values, tensor.shape.element_count());
    }
    check_cuda(cudaStreamSynchronize(nullptr), "synchronize weight QDQ");
  }
  result.seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started)
                       .count();
  return result;
}

}  // namespace gewell::weight_qdq
