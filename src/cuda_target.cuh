#pragma once

#include "gewell/console.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
namespace console = gewell::console;
__global__ void report_compiled_cuda_arch(int* result) {
#if defined(__CUDA_ARCH__)
  *result = __CUDA_ARCH__;
#endif
}

void require_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    require_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
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

  void FreeChecked() {
    if (pointer_ != nullptr) {
      void* released = pointer_;
      pointer_ = nullptr;
      bytes_ = 0;
      require_cuda(cudaFree(released), "cudaFree");
    }
  }

 private:
  void* pointer_{nullptr};
  std::size_t bytes_{};
};

bool validate_cuda_target(bool required) {
  constexpr std::string_view configured_arch = GEWELL_CUDA_ARCHITECTURE;
  if (configured_arch != "120a-real") {
    console::message(std::string("configured CUDA architecture is ") +
                     std::string(configured_arch) + ", expected 120a-real", true);
    return false;
  }

  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice ||
      (count_status == cudaSuccess && device_count == 0)) {
    const std::string reason = "no device";
    if (required) {
      console::message(std::string("CUDA device required: ") + std::string(reason), true);
      return false;
    }
    console::message("CUDA device check skipped: " + reason);
    return true;
  }
  if (count_status != cudaSuccess) {
    console::message(std::string("cudaGetDeviceCount failed: ") +
                     std::string(cudaGetErrorString(count_status)), true);
    return false;
  }

  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    console::message("could not query CUDA device 0", true);
    return false;
  }
  if (properties.major != 12 || properties.minor != 0) {
    console::message(std::string("CUDA device 0 is compute capability ") +
                     std::to_string(properties.major) + "." + std::to_string(properties.minor) +
                     ", expected 12.0", true);
    return false;
  }

  DeviceAllocation device_arch(sizeof(int));
  int compiled_arch = 0;
  report_compiled_cuda_arch<<<1, 1>>>(static_cast<int*>(device_arch.data()));
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    console::message(std::string("CUDA architecture probe launch failed: ") +
                     std::string(cudaGetErrorString(launch_status)), true);
    return false;
  }
  const cudaError_t copy_status =
      cudaMemcpy(&compiled_arch, device_arch.data(), sizeof(compiled_arch),
                 cudaMemcpyDeviceToHost);
  if (copy_status != cudaSuccess) {
    console::message(std::string("CUDA architecture probe failed: ") +
                     std::string(cudaGetErrorString(copy_status)), true);
    return false;
  }
  if (compiled_arch != 1'200) {
    console::message(std::string("kernel was compiled for sm_") + std::to_string(compiled_arch) +
                     ", expected sm_120", true);
    return false;
  }

  cublasLtHandle_t handle = nullptr;
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    console::message("cublasLtCreate failed", true);
    return false;
  }
  cublasLtDestroy(handle);

  console::section("GPU");
  console::field("cuda_device", properties.name);
  console::field("cuda_compute_capability", "12.0");
  console::field("cuda_target_validation", "passed");
  return true;
}

}  // namespace
