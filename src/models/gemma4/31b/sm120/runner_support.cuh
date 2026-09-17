#pragma once

#include "models/gemma4/31b/sm120/weights.cuh"
#include "models/gemma4/31b/sm120/execution_types.h"

#ifndef GEWELL_CUDA_ARCHITECTURE
#error "GEWELL_CUDA_ARCHITECTURE must name the configured CUDA target"
#endif

namespace gewell::gemma4_31b::sm120 {
inline void write_exclusive(const std::filesystem::path& path, const void* data,
                     std::size_t bytes) {
  const int descriptor =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0) {
    fail("create output file", path.string() + ": " + std::strerror(errno));
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
      ::unlink(path.c_str());
      fail("write output file", detail);
    }
    written += static_cast<std::size_t>(result);
  }
  if (::close(descriptor) != 0) {
    const std::string detail = path.string() + ": " + std::strerror(errno);
    ::unlink(path.c_str());
    fail("close output file", detail);
  }
}


inline void validate_cuda_device() {
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

inline artifact::Digest sha256_bytes(const void* data, std::size_t bytes) {
  artifact::Digest digest{};
  unsigned int digest_bytes = 0;
  if (EVP_Digest(data, bytes, digest.data(), &digest_bytes, EVP_sha256(),
                 nullptr) != 1 ||
      digest_bytes != digest.size()) {
    fail("graph-decode output SHA-256", "OpenSSL EVP_Digest failed");
  }
  return digest;
}



}  // namespace gewell::gemma4_31b::sm120
