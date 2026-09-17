#include "gewell/vision_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gewell::vision_primitives::BFloat16;

[[noreturn]] void fail(std::string_view label, std::string_view detail) {
  throw std::runtime_error(std::string(label) + ": " + std::string(detail));
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          elements * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  void clear() {
    check_cuda(cudaMemset(pointer_, 0, elements_ * sizeof(T)), "cudaMemset");
  }
  void copy_from(const std::vector<T>& values, std::size_t offset = 0) {
    if (offset > elements_ || values.size() > elements_ - offset) {
      fail("copy_from", "out-of-range copy");
    }
    check_cuda(cudaMemcpy(pointer_ + offset, values.data(),
                          values.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");
  }
  std::vector<T> copy_to_host() const {
    std::vector<T> result(elements_);
    check_cuda(cudaMemcpy(result.data(), pointer_, elements_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");
    return result;
  }

 private:
  T* pointer_{};
  std::size_t elements_{};
};

BFloat16 bf16(float value) { return __float2bfloat16_rn(value); }
float fp32(BFloat16 value) { return __bfloat162float(value); }

std::uint16_t bits(BFloat16 value) {
  std::uint16_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void require_bf16_equal(const std::vector<BFloat16>& actual,
                        const std::vector<BFloat16>& expected,
                        std::string_view label) {
  if (actual.size() != expected.size()) {
    fail(label, "size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (bits(actual[i]) != bits(expected[i])) {
      fail(label, "BF16 mismatch at " + std::to_string(i));
    }
  }
}

void require_close(float actual, float expected, float tolerance,
                   std::string_view label) {
  if (std::abs(actual - expected) > tolerance) {
    fail(label, "difference at value " + std::to_string(actual) +
                    " versus " + std::to_string(expected));
  }
}

void test_normalize() {
  constexpr std::size_t kElements = gewell::gemma4_31b::kVisionPatchWidth;
  std::vector<float> input(kElements);
  std::vector<BFloat16> expected(kElements);
  for (std::size_t i = 0; i < kElements; ++i) {
    input[i] = static_cast<float>(i % 17) / 16.0F;
    expected[i] = bf16(2.0F * (input[i] - 0.5F));
  }
  DeviceBuffer<float> device_input(kElements);
  DeviceBuffer<BFloat16> device_output(kElements);
  device_input.copy_from(input);
  gewell::vision_primitives::normalize_patch_values(
      device_input.get(), device_output.get(), 1);
  require_bf16_equal(device_output.copy_to_host(), expected, "normalize");
}

void test_position_add_in_place() {
  constexpr std::size_t kHidden = gewell::gemma4_31b::kVisionHiddenSize;
  constexpr std::size_t kPlane =
      static_cast<std::size_t>(gewell::gemma4_31b::kVisionPositionCount) *
      kHidden;
  std::vector<BFloat16> projected(kHidden);
  std::vector<BFloat16> x(kHidden);
  std::vector<BFloat16> y(kHidden);
  std::vector<BFloat16> expected(kHidden);
  for (std::size_t i = 0; i < kHidden; ++i) {
    projected[i] = bf16(static_cast<float>(static_cast<int>(i % 11) - 5) /
                        16.0F);
    x[i] = bf16(static_cast<float>(i % 7) / 32.0F);
    y[i] = bf16(-static_cast<float>(i % 5) / 64.0F);
    const BFloat16 position = bf16(fp32(x[i]) + fp32(y[i]));
    expected[i] = bf16(fp32(projected[i]) + fp32(position));
  }
  DeviceBuffer<BFloat16> device_values(kHidden);
  DeviceBuffer<std::int32_t> device_positions(2);
  DeviceBuffer<BFloat16> device_table(2 * kPlane);
  device_values.copy_from(projected);
  device_positions.copy_from(std::vector<std::int32_t>{1, 2});
  device_table.clear();
  device_table.copy_from(x, kHidden);
  device_table.copy_from(y, kPlane + 2 * kHidden);
  gewell::vision_primitives::add_patch_position_embeddings(
      device_values.get(), device_positions.get(), device_table.get(),
      device_values.get(), 1);
  require_bf16_equal(device_values.copy_to_host(), expected, "position add");
}

void test_rope_and_transposes() {
  constexpr std::uint32_t kRows = 2;
  constexpr std::uint32_t kHeads = gewell::gemma4_31b::kVisionHeadCount;
  constexpr std::uint32_t kHead = gewell::gemma4_31b::kVisionHeadSize;
  constexpr std::size_t kElements =
      static_cast<std::size_t>(kRows) * kHeads * kHead;
  std::vector<BFloat16> input(kElements);
  for (std::size_t i = 0; i < kElements; ++i) {
    input[i] = bf16(static_cast<float>(static_cast<int>(i % 23) - 11) /
                    32.0F);
  }
  DeviceBuffer<BFloat16> device_input(kElements);
  DeviceBuffer<BFloat16> device_head_major(kElements);
  DeviceBuffer<BFloat16> device_roundtrip(kElements);
  DeviceBuffer<std::int32_t> device_positions(4);
  device_input.copy_from(input);
  device_positions.copy_from(std::vector<std::int32_t>{0, 0, 1, 2});

  gewell::vision_primitives::token_heads_to_head_tokens(
      device_input.get(), device_head_major.get(), kRows);
  gewell::vision_primitives::head_tokens_to_token_heads(
      device_head_major.get(), device_roundtrip.get(), kRows);
  require_bf16_equal(device_roundtrip.copy_to_host(), input,
                     "transpose roundtrip");

  gewell::vision_primitives::apply_2d_rope_transpose(
      device_input.get(), device_positions.get(), device_head_major.get(),
      kRows);
  const std::vector<BFloat16> actual = device_head_major.copy_to_host();
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t dimension = 0; dimension < kHead; ++dimension) {
      const std::size_t source =
          (static_cast<std::size_t>(head)) * kHead + dimension;
      const std::size_t destination =
          (static_cast<std::size_t>(head) * kRows) * kHead + dimension;
      if (bits(actual[destination]) != bits(input[source])) {
        fail("rope", "position-zero transpose was not exact");
      }
    }
  }

  const std::uint32_t dimension = 7;
  const std::uint32_t paired = dimension + 18;
  const std::size_t row = static_cast<std::size_t>(kHeads) * kHead;
  const float inverse =
      1.0F / std::pow(100.0F, static_cast<float>(2 * dimension) / 36.0F);
  const BFloat16 cosine = bf16(std::cos(inverse));
  const BFloat16 sine = bf16(std::sin(inverse));
  const BFloat16 direct = bf16(fp32(input[row + dimension]) * fp32(cosine));
  const BFloat16 rotated =
      bf16(-fp32(input[row + paired]) * fp32(sine));
  const BFloat16 expected = bf16(fp32(direct) + fp32(rotated));
  const std::size_t destination = kHead + dimension;
  require_close(fp32(actual[destination]), fp32(expected), 0.004F, "rope");

  // Transformers constructs inverse frequencies on CPU. At this position,
  // recomputing frequency 2 with device powf changes the BF16 cosine by one
  // bit, so keep this exact regression for the pinned constant table.
  std::fill(input.begin(), input.end(), bf16(0.0F));
  input[2] = bf16(1.0F);
  device_input.copy_from(input);
  device_positions.copy_from(std::vector<std::int32_t>{76, 0, 0, 0});
  gewell::vision_primitives::apply_2d_rope_transpose(
      device_input.get(), device_positions.get(), device_head_major.get(), 1);
  const std::vector<BFloat16> pinned = device_head_major.copy_to_host();
  if (bits(pinned[2]) != 0xbbfcU) {
    fail("rope", "pinned high-position cosine differs");
  }
}

void test_softmax_in_place() {
  std::vector<BFloat16> scores{bf16(-1.0F), bf16(0.0F), bf16(1.0F),
                               bf16(4.0F),  bf16(4.0F), bf16(2.0F)};
  DeviceBuffer<BFloat16> device(6);
  device.copy_from(scores);
  gewell::vision_primitives::softmax_rows(device.get(), device.get(), 2, 3);
  const std::vector<BFloat16> actual = device.copy_to_host();
  for (std::uint32_t row = 0; row < 2; ++row) {
    float maximum = -INFINITY;
    for (std::uint32_t column = 0; column < 3; ++column) {
      maximum = std::max(maximum, fp32(scores[row * 3 + column]));
    }
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < 3; ++column) {
      sum += std::exp(fp32(scores[row * 3 + column]) - maximum);
    }
    for (std::uint32_t column = 0; column < 3; ++column) {
      const BFloat16 expected = bf16(
          std::exp(fp32(scores[row * 3 + column]) - maximum) / sum);
      require_close(fp32(actual[row * 3 + column]), fp32(expected), 0.002F,
                    "softmax");
    }
  }
}

void test_pool_and_standardize() {
  constexpr std::size_t kHidden = gewell::gemma4_31b::kVisionHiddenSize;
  constexpr std::uint32_t kPatchRows = 9;
  std::vector<BFloat16> input(kPatchRows * kHidden);
  std::vector<std::int32_t> positions(kPatchRows * 2);
  // Keep the final row at (2,2), which is the width-bearing ABI position, but
  // shuffle the preceding coordinates to prove pooling follows positions.
  const std::int32_t coordinates[9][2] = {
      {1, 1}, {0, 0}, {2, 1}, {1, 0}, {0, 2},
      {2, 0}, {0, 1}, {1, 2}, {2, 2},
  };
  for (std::uint32_t row = 0; row < kPatchRows; ++row) {
    positions[row * 2] = coordinates[row][0];
    positions[row * 2 + 1] = coordinates[row][1];
    for (std::size_t dimension = 0; dimension < kHidden; ++dimension) {
      input[static_cast<std::size_t>(row) * kHidden + dimension] =
          bf16(static_cast<float>(coordinates[row][0] +
                                  3 * coordinates[row][1]) +
               static_cast<float>(dimension % 7) / 16.0F);
    }
  }
  DeviceBuffer<BFloat16> device_input(input.size());
  DeviceBuffer<std::int32_t> device_positions(positions.size());
  DeviceBuffer<float> device_pooled(kHidden);
  device_input.copy_from(input);
  device_positions.copy_from(positions);
  gewell::vision_primitives::pool_3x3_scaled(
      device_input.get(), device_positions.get(), device_pooled.get(),
      kPatchRows);
  const std::vector<float> pooled = device_pooled.copy_to_host();
  constexpr float kWeight = 1.0F / 9.0F;
  constexpr float kRootHidden = 33.941125496954281F;
  for (std::size_t dimension = 0; dimension < kHidden; ++dimension) {
    float average = 0.0F;
    for (std::uint32_t slot = 0; slot < 9; ++slot) {
      std::uint32_t source = 0;
      while (coordinates[source][0] != static_cast<std::int32_t>(slot % 3) ||
             coordinates[source][1] != static_cast<std::int32_t>(slot / 3)) {
        ++source;
      }
      average = std::fma(
          kWeight,
          fp32(input[static_cast<std::size_t>(source) * kHidden + dimension]),
          average);
    }
    const float expected = fp32(bf16(average)) * kRootHidden;
    require_close(pooled[dimension], expected, 0.0001F, "pool");
  }

  std::vector<BFloat16> bias(kHidden);
  std::vector<BFloat16> scale(kHidden);
  std::vector<BFloat16> expected(kHidden);
  for (std::size_t dimension = 0; dimension < kHidden; ++dimension) {
    bias[dimension] = bf16(static_cast<float>(dimension % 5));
    scale[dimension] = bf16(0.25F + static_cast<float>(dimension % 3) / 8.0F);
    expected[dimension] =
        bf16((pooled[dimension] - fp32(bias[dimension])) *
             fp32(scale[dimension]));
  }
  DeviceBuffer<BFloat16> device_bias(kHidden);
  DeviceBuffer<BFloat16> device_scale(kHidden);
  DeviceBuffer<BFloat16> device_output(kHidden);
  device_bias.copy_from(bias);
  device_scale.copy_from(scale);
  gewell::vision_primitives::standardize(
      device_pooled.get(), device_bias.get(), device_scale.get(),
      device_output.get(), 1);
  require_bf16_equal(device_output.copy_to_host(), expected, "standardize");
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      fail("vision primitives", "no CUDA device");
    }
    test_normalize();
    test_position_add_in_place();
    test_rope_and_transposes();
    test_softmax_in_place();
    test_pool_and_standardize();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    std::cout << "vision primitives: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vision primitives: FAIL: " << error.what() << '\n';
    return 1;
  }
}
