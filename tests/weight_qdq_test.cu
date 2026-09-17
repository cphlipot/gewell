#include "gewell/weight_qdq.h"
#include "../src/weight_qdq_detail.cuh"

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
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <iostream>

namespace {
using namespace gewell::weight_qdq;
using namespace gewell::weight_qdq::detail;
namespace model = gewell::gemma4_31b;
namespace artifact = gewell::artifact;

[[nodiscard]] std::uint16_t bf16_bits(BFloat16 value) {
  return static_cast<__nv_bfloat16_raw>(value).x;
}

[[nodiscard]] BFloat16 bf16_from_bits(std::uint16_t bits) {
  BFloat16 value;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::vector<BFloat16> bf16_from_words(
    std::initializer_list<std::uint16_t> words) {
  std::vector<BFloat16> values;
  values.reserve(words.size());
  for (const std::uint16_t word : words) {
    values.push_back(bf16_from_bits(word));
  }
  return values;
}

void fp8_qdq_host(std::vector<BFloat16>& values) {
  float amax = 0.0F;
  for (const BFloat16 value : values) {
    amax = std::max(amax, std::fabs(__bfloat162float(value)));
  }
  if (amax == 0.0F) {
    return;
  }
  const float scale = amax / kE4m3Maximum;
  for (BFloat16& value : values) {
    value = round_bf16(
        round_e4m3(__bfloat162float(value) / scale) * scale);
  }
}

void nvfp4_qdq_host(std::vector<BFloat16>& values) {
  if (values.size() % 16 != 0) {
    fail("host NVFP4 QDQ input is not divisible by 16");
  }
  float amax = 0.0F;
  for (const BFloat16 value : values) {
    amax = std::max(amax, std::fabs(__bfloat162float(value)));
  }
  if (amax == 0.0F) {
    return;
  }
  const float global_scale =
      amax / (kE2m1Maximum * kE4m3Maximum);
  for (std::size_t base = 0; base < values.size(); base += 16) {
    float block_amax = 0.0F;
    for (std::size_t offset = 0; offset < 16; ++offset) {
      block_amax = std::max(
          block_amax, std::fabs(__bfloat162float(values[base + offset])));
    }
    const float block_scale =
        nvfp4_block_scale(block_amax, global_scale);
    const float combined_scale = block_scale * global_scale;
    for (std::size_t offset = 0; offset < 16; ++offset) {
      const float value = __bfloat162float(values[base + offset]);
      values[base + offset] =
          round_bf16(round_e2m1(value / combined_scale) * combined_scale);
    }
  }
}

[[nodiscard]] bool same_bf16(const std::vector<BFloat16>& first,
                             const std::vector<BFloat16>& second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (bf16_bits(first[index]) != bf16_bits(second[index])) {
      return false;
    }
  }
  return true;
}

void expect(bool condition, std::string_view message) {
  if (!condition) {
    fail(std::string(message));
  }
}

void expect_bf16_words(const std::vector<BFloat16>& values,
                       const std::vector<std::uint16_t>& expected,
                       std::string_view message) {
  expect(values.size() == expected.size(), message);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (bf16_bits(values[index]) != expected[index]) {
      fail(std::string(message) + " at element " + std::to_string(index) +
           ": expected 0x" + [&] {
             std::ostringstream output;
             output << std::hex << expected[index];
             return output.str();
           }() +
           ", got 0x" + [&] {
             std::ostringstream output;
             output << std::hex << bf16_bits(values[index]);
             return output.str();
           }());
    }
  }
}

template <typename Function>
void expect_failure(Function&& function, std::string_view needle) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find(needle) != std::string_view::npos) {
      return;
    }
    fail(std::string("unexpected failure: ") + error.what());
  }
  fail(std::string("expected failure containing: ") + std::string(needle));
}

bool run_self_tests(std::string* failure) {
  try {
    const Mask default_mask;
    expect(default_mask.all_bf16(), "default QDQ mask is not all BF16");
    const SelectionSummary default_summary = summarize(default_mask);
    expect(default_summary.bf16.tensor_count == 410,
           "default QDQ projection count mismatch");
    expect(default_summary.bf16.source_bf16_bytes == 58'573'455'360ULL,
           "default QDQ projection byte count mismatch");
    expect(default_summary.fp8.tensor_count == 0 &&
               default_summary.nvfp4.tensor_count == 0,
           "default QDQ mask selects a quantized projection");
    const artifact::ArtifactFile empty_artifact;
    const ApplySummary no_op =
        apply_in_place(default_mask, empty_artifact, nullptr);
    expect(no_op.seconds == 0.0 && no_op.selection.bf16.tensor_count == 410,
           "all-BF16 QDQ did not remain a device-free no-op");

    const Mask overrides = Mask::Parse(
        "# family defaults and one exception\n"
        "* gate_proj nvfp4\n"
        "* up_proj nvfp4\n"
        "* down_proj nvfp4\n"
        "17 gate_proj fp8\n",
        "override-mask");
    expect(overrides.type_for(0, model::TensorRole::gate_proj) == Type::nvfp4,
           "wildcard QDQ rule did not apply");
    expect(overrides.type_for(17, model::TensorRole::gate_proj) == Type::fp8,
           "later QDQ override did not win");
    expect(overrides.type_for(17, model::TensorRole::q_proj) == Type::bf16,
           "unspecified projection did not remain BF16");
    expect(overrides.type_for(
               model::kPhysicalTensors[model::kVisionProjectionPhysicalId]) ==
               Type::bf16,
           "text QDQ mask selected the vision projection");
    expect(overrides.type_for(model::kPhysicalTensors[
               model::kVisionLayerWeightsFirstPhysicalId + 1]) == Type::bf16,
           "text QDQ mask selected a vision-layer projection");
    const SelectionSummary override_summary = summarize(overrides);
    expect(override_summary.nvfp4.tensor_count == 179 &&
               override_summary.fp8.tensor_count == 1 &&
               override_summary.bf16.tensor_count == 230,
           "MLP QDQ selection counts mismatch");
    expect(override_summary.nvfp4.source_bf16_bytes +
                   override_summary.fp8.source_bf16_bytes ==
               41'617'981'440ULL,
           "MLP QDQ selection byte count mismatch");

    const Mask native = Mask::Parse(
        "* gate_proj nvfp4_w4a4\n* up_proj nvfp4_w4a4\n"
        "* down_proj nvfp4_w4a4\n17 gate_proj bf16\n", "native-mask");
    expect(summarize(native).nvfp4_w4a4.tensor_count == 179 &&
               summarize(native).bf16.tensor_count == 231 && !native.has_qdq(),
           "native mask must remain independent from weight QDQ");
    const Mask native_attention = Mask::Parse(
        "* q_proj nvfp4_w4a4\n* k_proj nvfp4_w4a4\n"
        "* v_proj nvfp4_w4a4\n* o_proj nvfp4_w4a4\n", "native-attention-mask");
    expect(summarize(native_attention).nvfp4_w4a4.tensor_count == 230 &&
               !native_attention.has_qdq(), "NVFP4 attention selection mismatch");
    expect_failure([] { Mask::Parse("5 v_proj nvfp4_w4a4\n"); },
                   "global layers do not have v_proj");

    const Mask native_fp8 = Mask::Parse(
        "* q_proj fp8_w8a8\n* v_proj fp8_w8a8\n"
        "* gate_proj nvfp4_w4a4\n17 gate_proj fp8_w8a8\n", "mixed-mask");
    expect(summarize(native_fp8).fp8_w8a8.tensor_count == 111 &&
               summarize(native_fp8).nvfp4_w4a4.tensor_count == 59 &&
               !native_fp8.has_qdq(), "native mixed projection selection mismatch");

    const Mask local_v = Mask::Parse("* v_proj fp8\n", "local-v-mask");
    expect(summarize(local_v).fp8.tensor_count == 50,
           "wildcard v_proj did not select exactly the local layers");
    expect_failure(
        [&] { (void)local_v.type_for(5, model::TensorRole::v_proj); },
        "nonexistent global v_proj");

    const Mask cleared = Mask::Parse(
        "* q_proj fp8\n17 q_proj bf16\n", "cleared-override-mask");
    expect(summarize(cleared).fp8.tensor_count == 59,
           "later BF16 rule did not clear a wildcard rule");

    const Mask every_projection = Mask::Parse(
        "0 q_proj fp8\n"
        "0 k_proj fp8\n"
        "0 v_proj fp8\n"
        "0 o_proj fp8\n"
        "0 gate_proj fp8\n"
        "0 up_proj fp8\n"
        "0 down_proj fp8\n",
        "all-projections-mask");
    expect(summarize(every_projection).fp8.tensor_count == 7,
           "one or more projection families did not map independently");

    std::uint32_t selectable_tensors = 0;
    for (const model::TensorSpec& tensor : model::kPhysicalTensors) {
      if (!projection_index(tensor.role).has_value()) {
        continue;
      }
      ++selectable_tensors;
      expect(tensor.layer >= 0 && tensor.layer < model::kLayerCount &&
                 tensor.shape.rank == 2 &&
                 tensor.shape.dimensions[1] % 16 == 0,
             "compiled projection violates the QDQ shape contract");
    }
    expect(selectable_tensors == 410,
           "compiled selectable projection catalog count mismatch");

    const model::TensorSpec& embedding = model::kPhysicalTensors[0];
    expect(overrides.type_for(embedding) == Type::bf16,
           "embedding was selected for QDQ");
    expect(overrides.type_for(0, model::TensorRole::input_norm) == Type::bf16 &&
               overrides.type_for(0, model::TensorRole::layer_scalar) ==
                   Type::bf16,
           "non-projection weight was selected for QDQ");
    expect_failure(
        [&] { (void)default_mask.type_for(60, model::TensorRole::q_proj); },
        "outside 0..59");

    expect_failure(
        [] { Mask::Parse("5 v_proj fp8\n", "bad-global-v"); },
        "global layers do not have v_proj");
    expect_failure([] { Mask::Parse("60 q_proj fp8\n", "bad-layer"); },
                   "0..59");
    expect_failure([] { Mask::Parse("-1 q_proj fp8\n", "bad-layer"); },
                   "0..59");
    expect_failure([] { Mask::Parse("0 query fp8\n", "bad-role"); },
                   "unknown projection");
    expect_failure([] { Mask::Parse("0 q_proj int8\n", "bad-type"); },
                   "unknown QDQ type");
    expect_failure([] { Mask::Parse("0 q_proj\n", "short-line"); },
                   "expected exactly");
    expect_failure(
        [] { Mask::Parse("0 q_proj fp8 trailing\n", "long-line"); },
        "expected exactly");
    expect(Mask::Parse(std::string(kMaximumMaskBytes, '#'), "maximum-mask")
               .all_bf16(),
           "exactly 1 MiB QDQ mask did not parse");
    expect_failure(
        [] {
          Mask::Parse(std::string(kMaximumMaskBytes + 1, '#'),
                      "oversized-mask");
        },
        "exceeds 1 MiB");

    expect(round_e4m3(448.0F) == 448.0F &&
               round_e4m3(500.0F) == 448.0F &&
               round_e4m3(0x1p-10F) == 0.0F &&
               round_e4m3(0x1.8p-9F) == 0x1p-8F &&
               round_e4m3(1.0625F) == 1.0F &&
               round_e4m3(1.1875F) == 1.25F,
           "E4M3 saturation or round-to-nearest-even mismatch");
    expect(round_e2m1(0.25F) == 0.0F &&
               round_e2m1(0.75F) == 1.0F &&
               round_e2m1(1.25F) == 1.0F &&
               round_e2m1(1.75F) == 2.0F &&
               round_e2m1(2.5F) == 2.0F &&
               round_e2m1(3.5F) == 4.0F &&
               round_e2m1(5.0F) == 4.0F,
           "E2M1 round-to-nearest-even mismatch");
    expect(nvfp4_block_scale(0.0F, 1.0F) == 1.0F &&
               nvfp4_block_scale(0x1p-20F, 1.0F) ==
                   kNvfp4MinimumBlockScale &&
               nvfp4_block_scale(6.375F, 1.0F) == 1.0F &&
               nvfp4_block_scale(7.125F, 1.0F) == 1.25F,
           "NVFP4 block-scale clamp or rounding mismatch");

    std::vector<BFloat16> fp8_golden = bf16_from_words(
        {0x43e0, 0xc3e0, 0x3a80, 0xba80, 0x3a81, 0xba81,
         0x3f88, 0xbf88, 0x3f98, 0xbf98, 0x3fa8, 0xbfa8,
         0x3fb8, 0xbfb8, 0x0000, 0x8000});
    fp8_qdq_host(fp8_golden);
    expect_bf16_words(
        fp8_golden,
        {0x43e0, 0xc3e0, 0x0000, 0x8000, 0x3b00, 0xbb00,
         0x3f80, 0xbf80, 0x3fa0, 0xbfa0, 0x3fa0, 0xbfa0,
         0x3fc0, 0xbfc0, 0x0000, 0x8000},
        "static FP8 QDQ golden mismatch");

    std::vector<BFloat16> nvfp4_golden(96, bf16_from_bits(0x0000));
    nvfp4_golden[0] = bf16_from_bits(0x4528);
    nvfp4_golden[1] = bf16_from_bits(0xc528);
    const std::array<std::uint16_t, 16> nvfp4_tie_inputs{
        0x3e80, 0xbe80, 0x3f40, 0xbf40, 0x3fa0, 0xbfa0,
        0x3fe0, 0xbfe0, 0x4020, 0xc020, 0x4060, 0xc060,
        0x40a0, 0xc0a0, 0x40c0, 0xc0c0};
    for (std::size_t index = 0; index < nvfp4_tie_inputs.size(); ++index) {
      nvfp4_golden[16 + index] = bf16_from_bits(nvfp4_tie_inputs[index]);
    }
    nvfp4_golden[32] = bf16_from_bits(0x40cc);
    nvfp4_golden[33] = bf16_from_bits(0xc0cc);
    nvfp4_golden[48] = bf16_from_bits(0x40e4);
    nvfp4_golden[49] = bf16_from_bits(0xc0e4);
    nvfp4_golden[64] = bf16_from_bits(0x3ac0);
    nvfp4_golden[65] = bf16_from_bits(0xbac0);
    nvfp4_qdq_host(nvfp4_golden);
    std::vector<std::uint16_t> nvfp4_expected(96, 0x0000);
    nvfp4_expected[0] = 0x4528;
    nvfp4_expected[1] = 0xc528;
    const std::array<std::uint16_t, 16> nvfp4_tie_expected{
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x3f80, 0xbf80,
        0x4000, 0xc000, 0x4000, 0xc000, 0x4080, 0xc080,
        0x4080, 0xc080, 0x40c0, 0xc0c0};
    for (std::size_t index = 0; index < nvfp4_tie_expected.size(); ++index) {
      nvfp4_expected[16 + index] = nvfp4_tie_expected[index];
    }
    nvfp4_expected[32] = 0x40c0;
    nvfp4_expected[33] = 0xc0c0;
    nvfp4_expected[48] = 0x40f0;
    nvfp4_expected[49] = 0xc0f0;
    nvfp4_expected[64] = 0x3b00;
    nvfp4_expected[65] = 0xbb00;
    expect_bf16_words(nvfp4_golden, nvfp4_expected,
                      "NVFP4 QDQ golden mismatch");

    std::vector<BFloat16> fp8_values;
    std::vector<BFloat16> nvfp4_values;
    fp8_values.reserve(32);
    nvfp4_values.reserve(32);
    for (int index = 0; index < 32; ++index) {
      const float value =
          static_cast<float>((index * 17) % 29 - 14) * 0.1875F;
      fp8_values.push_back(round_bf16(value));
      nvfp4_values.push_back(round_bf16(value));
    }
    fp8_qdq_host(fp8_values);
    nvfp4_qdq_host(nvfp4_values);
    for (const BFloat16 value : fp8_values) {
      expect(std::isfinite(__bfloat162float(value)),
             "host FP8 QDQ produced a non-finite value");
    }
    for (const BFloat16 value : nvfp4_values) {
      expect(std::isfinite(__bfloat162float(value)),
             "host NVFP4 QDQ produced a non-finite value");
    }
    std::vector<BFloat16> zeros(16, round_bf16(0.0F));
    const std::vector<BFloat16> original_zeros = zeros;
    nvfp4_qdq_host(zeros);
    expect(same_bf16(zeros, original_zeros),
           "all-zero NVFP4 block was changed");
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

bool run_device_self_tests(std::string* failure) {
  try {
    auto run_on_device = [](Type type,
                            const std::vector<BFloat16>& source,
                            std::size_t offset,
                            std::size_t elements) {
      expect(offset <= source.size() && elements <= source.size() - offset,
             "QDQ device self-test range is invalid");
      BFloat16* device = nullptr;
      check_cuda(cudaMalloc(&device, source.size() * sizeof(BFloat16)),
                 "cudaMalloc QDQ self-test values");
      try {
        check_cuda(cudaMemcpy(device, source.data(),
                              source.size() * sizeof(BFloat16),
                              cudaMemcpyHostToDevice),
                   "copy QDQ self-test input");
        DeviceState state;
        apply_tensor(state, type, device + offset, elements);
        check_cuda(cudaStreamSynchronize(nullptr),
                   "synchronize QDQ self-test");
        std::vector<BFloat16> actual(source.size());
        check_cuda(cudaMemcpy(actual.data(), device,
                              actual.size() * sizeof(BFloat16),
                              cudaMemcpyDeviceToHost),
                   "copy QDQ self-test output");
        check_cuda(cudaFree(device), "cudaFree QDQ self-test values");
        return actual;
      } catch (...) {
        cudaFree(device);
        throw;
      }
    };

    std::vector<BFloat16> source;
    source.reserve(40);
    for (int index = 0; index < 40; ++index) {
      const float value =
          static_cast<float>((index * 23) % 37 - 18) * 0.15625F;
      source.push_back(round_bf16(value));
    }

    auto run = [&](Type type) {
      std::vector<BFloat16> expected(source.begin() + 4, source.begin() + 36);
      if (type == Type::fp8) {
        fp8_qdq_host(expected);
      } else {
        nvfp4_qdq_host(expected);
      }

      const std::vector<BFloat16> actual =
          run_on_device(type, source, 4, expected.size());
      expect(same_bf16(
                 std::vector<BFloat16>(actual.begin() + 4,
                                       actual.begin() + 36),
                 expected),
             std::string(type_name(type)) +
                 " device QDQ differs from host reference");
      expect(same_bf16(
                 std::vector<BFloat16>(actual.begin(), actual.begin() + 4),
                 std::vector<BFloat16>(source.begin(), source.begin() + 4)) &&
                 same_bf16(std::vector<BFloat16>(actual.begin() + 36,
                                                 actual.end()),
                           std::vector<BFloat16>(source.begin() + 36,
                                                 source.end())),
             std::string(type_name(type)) +
                 " device QDQ changed neighboring values");
    };

    run(Type::fp8);
    run(Type::nvfp4);

    const std::vector<BFloat16> zeros(32, round_bf16(0.0F));
    expect(same_bf16(run_on_device(Type::fp8, zeros, 0, zeros.size()),
                     zeros) &&
               same_bf16(
                   run_on_device(Type::nvfp4, zeros, 0, zeros.size()), zeros),
           "all-zero device QDQ was not a no-op");

    std::vector<BFloat16> nonfinite(16, round_bf16(0.0F));
    nonfinite[7] = round_bf16(std::numeric_limits<float>::infinity());
    expect_failure(
        [&] {
          (void)run_on_device(Type::nvfp4, nonfinite, 0, nonfinite.size());
        },
        "non-finite BF16 value");

    constexpr std::size_t kCappedGridElements =
        static_cast<std::size_t>(kThreads) * kMaximumReductionBlocks;
    constexpr std::size_t kLargeElements = kCappedGridElements + 64;

    std::vector<BFloat16> large_fp8(kLargeElements,
                                    bf16_from_bits(0x0000));
    std::vector<BFloat16> large_fp8_expected = large_fp8;
    const auto set_fp8 = [&](std::size_t index, std::uint16_t input,
                             std::uint16_t expected) {
      large_fp8[index] = bf16_from_bits(input);
      large_fp8_expected[index] = bf16_from_bits(expected);
    };
    set_fp8(0, 0x43e0, 0x43e0);
    set_fp8(1, 0xc3e0, 0xc3e0);
    set_fp8(kCappedGridElements, 0x3a80, 0x0000);
    set_fp8(kCappedGridElements + 1, 0xba80, 0x8000);
    set_fp8(kCappedGridElements + 2, 0x3a81, 0x3b00);
    set_fp8(kCappedGridElements + 3, 0xba81, 0xbb00);
    set_fp8(kCappedGridElements + 16, 0x3f88, 0x3f80);
    set_fp8(kCappedGridElements + 17, 0xbf88, 0xbf80);
    set_fp8(kCappedGridElements + 18, 0x3f98, 0x3fa0);
    set_fp8(kCappedGridElements + 19, 0xbf98, 0xbfa0);
    expect(same_bf16(
               run_on_device(Type::fp8, large_fp8, 0, large_fp8.size()),
               large_fp8_expected),
           "FP8 device QDQ capped-grid golden mismatch");

    std::vector<BFloat16> large_nvfp4(kLargeElements,
                                      bf16_from_bits(0x0000));
    std::vector<BFloat16> large_nvfp4_expected = large_nvfp4;
    const auto set_nvfp4 = [&](std::size_t index, std::uint16_t input,
                               std::uint16_t expected) {
      large_nvfp4[index] = bf16_from_bits(input);
      large_nvfp4_expected[index] = bf16_from_bits(expected);
    };
    set_nvfp4(0, 0x4528, 0x4528);
    set_nvfp4(1, 0xc528, 0xc528);
    set_nvfp4(kCappedGridElements, 0x40cc, 0x40c0);
    set_nvfp4(kCappedGridElements + 1, 0xc0cc, 0xc0c0);
    set_nvfp4(kCappedGridElements + 16, 0x40e4, 0x40f0);
    set_nvfp4(kCappedGridElements + 17, 0xc0e4, 0xc0f0);
    set_nvfp4(kCappedGridElements + 32, 0x3ac0, 0x3b00);
    set_nvfp4(kCappedGridElements + 33, 0xbac0, 0xbb00);
    expect(same_bf16(run_on_device(Type::nvfp4, large_nvfp4, 0,
                                   large_nvfp4.size()),
                     large_nvfp4_expected),
           "NVFP4 device QDQ capped-grid golden mismatch");
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool device = argc == 2 && std::string_view(argv[1]) == "--device";
  if (argc > 1 && !device) {
    std::cerr << "usage: weight_qdq_test [--device]\n";
    return 2;
  }
  std::string failure;
  if (!(device ? run_device_self_tests(&failure) : run_self_tests(&failure))) {
    std::cerr << failure << "\n";
    return 1;
  }
  std::cout << (device ? "weight QDQ device tests: ok\n" : "weight QDQ host tests: ok\n");
  return 0;
}
