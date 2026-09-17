#include "gewell/replay_metrics.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::replay_metrics {
namespace {

constexpr unsigned kThreads = 256;

struct Extrema {
  double reference;
  double replay;
  std::uint32_t reference_id;
  std::uint32_t replay_id;
  std::uint32_t flags;
};

struct Sums {
  double reference_mass;
  double replay_mass;
  double reference_difference;
  double replay_difference;
};

__device__ void select_max(double value, std::uint32_t id, double& maximum,
                           std::uint32_t& maximum_id) {
  if (value > maximum || (value == maximum && id < maximum_id)) {
    maximum = value;
    maximum_id = id;
  }
}

__global__ void compare_kernel(const __nv_bfloat16* reference,
                               const __nv_bfloat16* replay,
                               const std::uint32_t* recorded_tokens,
                               std::uint32_t vocabulary_size, Row* output) {
  __shared__ Extrema extrema[kThreads];
  __shared__ Sums sums[kThreads];
  const std::uint32_t row = blockIdx.x;
  const unsigned lane = threadIdx.x;
  const std::size_t offset = static_cast<std::size_t>(row) * vocabulary_size;
  const std::uint32_t token = recorded_tokens[row];
  Extrema local{-CUDART_INF, -CUDART_INF, UINT_MAX, UINT_MAX, 0};
  for (std::uint32_t id = lane; id < vocabulary_size; id += kThreads) {
    const double a = __bfloat162float(reference[offset + id]);
    const double b = __bfloat162float(replay[offset + id]);
    if (!isfinite(a) || !isfinite(b)) {
      local.flags |= nonfinite;
    }
    if ((isfinite(a) && fabs(a) > 30.0) ||
        (isfinite(b) && fabs(b) > 30.0)) {
      local.flags |= outside_softcap;
    }
    select_max(a, id, local.reference, local.reference_id);
    select_max(b, id, local.replay, local.replay_id);
  }
  extrema[lane] = local;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride != 0; stride /= 2) {
    if (lane < stride) {
      const Extrema other = extrema[lane + stride];
      select_max(other.reference, other.reference_id,
                 extrema[lane].reference, extrema[lane].reference_id);
      select_max(other.replay, other.replay_id,
                 extrema[lane].replay, extrema[lane].replay_id);
      extrema[lane].flags |= other.flags;
    }
    __syncthreads();
  }
  const Extrema maxima = extrema[0];
  std::uint32_t flags = maxima.flags;
  if (token >= vocabulary_size) {
    flags |= invalid_recorded_token;
  } else if (token != maxima.reference_id) {
    flags |= reference_argmax_mismatch;
  }
  if (flags & (nonfinite | outside_softcap | invalid_recorded_token)) {
    if (lane == 0) {
      output[row] = {CUDART_NAN, CUDART_NAN, CUDART_NAN, CUDART_NAN,
                     maxima.reference_id, maxima.replay_id, token, flags};
    }
    return;
  }

  Sums partial{};
  for (std::uint32_t id = lane; id < vocabulary_size; id += kThreads) {
    const double a = static_cast<double>(
                         __bfloat162float(reference[offset + id])) -
                     maxima.reference;
    const double b = static_cast<double>(
                         __bfloat162float(replay[offset + id])) -
                     maxima.replay;
    const double pa = exp(a);
    const double pb = exp(b);
    partial.reference_mass += pa;
    partial.replay_mass += pb;
    partial.reference_difference += pa * (a - b);
    partial.replay_difference += pb * (b - a);
  }
  sums[lane] = partial;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride != 0; stride /= 2) {
    if (lane < stride) {
      sums[lane].reference_mass += sums[lane + stride].reference_mass;
      sums[lane].replay_mass += sums[lane + stride].replay_mass;
      sums[lane].reference_difference +=
          sums[lane + stride].reference_difference;
      sums[lane].replay_difference += sums[lane + stride].replay_difference;
    }
    __syncthreads();
  }
  if (lane == 0) {
    const Sums total = sums[0];
    const double normalization = log(total.replay_mass / total.reference_mass);
    double forward = total.reference_difference / total.reference_mass +
                     normalization;
    double reverse = total.replay_difference / total.replay_mass -
                     normalization;
    // Match the existing FP64 scorer's roundoff allowance at zero.
    if (forward < 0.0 && forward > -1e-12) forward = 0.0;
    if (reverse < 0.0 && reverse > -1e-12) reverse = 0.0;
    if (!isfinite(forward) || !isfinite(reverse) || forward < 0.0 ||
        reverse < 0.0) {
      flags |= invalid_kl;
    }
    output[row] = {
        forward, reverse,
        maxima.replay - __bfloat162float(replay[offset + token]),
        maxima.reference - __bfloat162float(reference[offset + token]),
        maxima.reference_id, maxima.replay_id, token, flags};
  }
}

}  // namespace

void compare_rows(const __nv_bfloat16* reference,
                  const __nv_bfloat16* replay,
                  const std::uint32_t* recorded_tokens,
                  std::uint32_t rows, std::uint32_t vocabulary_size,
                  Row* output, cudaStream_t stream) {
  if (reference == nullptr || replay == nullptr || recorded_tokens == nullptr ||
      output == nullptr || rows == 0 || rows > INT_MAX ||
      vocabulary_size == 0 || vocabulary_size > INT_MAX ||
      static_cast<std::size_t>(rows) >
          std::numeric_limits<std::size_t>::max() /
              (static_cast<std::size_t>(vocabulary_size) *
               sizeof(__nv_bfloat16))) {
    throw std::invalid_argument("replay metrics: invalid pointers or shape");
  }
  compare_kernel<<<rows, kThreads, 0, stream>>>(
      reference, replay, recorded_tokens, vocabulary_size, output);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("launch replay metrics: ") +
                             cudaGetErrorString(error));
  }
}

}  // namespace gewell::replay_metrics
