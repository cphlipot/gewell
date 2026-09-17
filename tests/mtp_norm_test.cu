#include "../src/models/gemma4/31b/sm120/mtp/norm.cuh"
#include "../src/models/gemma4/31b/sm120/mtp/cuda.cuh"
#include "gewell/bf16_primitives.h"

#include <iostream>
#include <vector>

namespace {
namespace norm = gewell::mtp_target::detail;
namespace p = gewell::bf16_primitives;
using BF16 = __nv_bfloat16;
using gewell::mtp_cuda::Buffer;
using gewell::mtp_cuda::check;
constexpr unsigned kHidden = gewell::gemma4_31b::kHiddenSize;

__global__ void fill(BF16* values, unsigned count, unsigned tag) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  unsigned hash = (i + tag * 797) * 2654435761U;
  hash ^= hash >> 13;
  const float scale = i % 13 == 0 ? 32.0F : i % 17 == 0 ? 0.00001F : 0.125F;
  values[i] = __float2bfloat16_rn((int(hash & 255) - 127) * scale);
}

void require_equal(const Buffer& actual, const Buffer& expected) {
  std::vector<unsigned short> a(actual.size() / 2), b(a.size());
  check(cudaMemcpy(a.data(), actual.data(), actual.size(), cudaMemcpyDeviceToHost),
        "read fused normalization");
  check(cudaMemcpy(b.data(), expected.data(), expected.size(), cudaMemcpyDeviceToHost),
        "read separate normalization");
  if (a != b) throw std::runtime_error("fused normalization differs from separate BF16 operations");
}

template <bool Scale>
void test(unsigned rows) {
  const unsigned count = rows * kHidden;
  Buffer original(count * 2), branch(count * 2), residual(count * 2),
      expected(count * 2), temporary(count * 2), output(count * 2),
      expected_output(count * 2), post(kHidden * 2), next(kHidden * 2), scalar(2);
  const auto initialize = [](Buffer& buffer, unsigned tag) {
    const unsigned count = buffer.size() / 2;
    fill<<<(count + 255) / 256, 256>>>(buffer.at<BF16>(0), count, tag);
    check(cudaGetLastError(), "initialize normalization fixture");
  };
  initialize(original, 1);
  initialize(residual, 2);
  initialize(post, 3);
  initialize(next, 4);
  initialize(scalar, 5);

  // The independently executed primitive chain defines every rounding boundary
  // and the reduction order, including rows with large and tiny components.
  p::rms_norm(original.at<BF16>(0), post.at<BF16>(0), temporary.at<BF16>(0),
              rows, kHidden);
  p::residual_add(residual.at<BF16>(0), temporary.at<BF16>(0),
                  expected.at<BF16>(0), count);
  if constexpr (Scale)
    p::trained_scalar(expected.at<BF16>(0), scalar.at<BF16>(0), count);
  p::rms_norm(expected.at<BF16>(0), next.at<BF16>(0), expected_output.at<BF16>(0),
              rows, kHidden);
  check(cudaMemcpy(branch.data(), original.data(), original.size(), cudaMemcpyDeviceToDevice),
        "copy normalization input");
  norm::residual_norm<Scale><<<rows, norm::kNormThreads>>>(
      branch.at<BF16>(0), post.at<BF16>(0), residual.at<BF16>(0), scalar.at<BF16>(0),
      next.at<BF16>(0), output.at<BF16>(0));
  check(cudaGetLastError(), "fused normalization launch");
  require_equal(branch, expected);
  require_equal(output, expected_output);
  std::cout << "rows=" << rows << " scale=" << Scale
            << " residual=exact normalized=exact\n";
}
}  // namespace

int main() {
  try {
    for (unsigned rows : {1U, 2U, 3U, 4U, 5U, 17U, 1280U}) {
      test<false>(rows);
      test<true>(rows);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
