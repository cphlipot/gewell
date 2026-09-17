#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gewell::mtp_cuda {
inline void check(cudaError_t value, const char* operation) {
  if (value != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(value));
}

inline void check(cublasStatus_t value, const char* operation) {
  if (value != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                             std::to_string(value));
}

inline std::size_t align(std::size_t n) { return (n + 255) & ~std::size_t(255); }

class Buffer {
 public:
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes) check(cudaMalloc(&data_, bytes), "MTP scratch allocation");
  }
  ~Buffer() {
    if (data_) cudaFree(data_);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* data() const { return data_; }
  std::size_t size() const { return bytes_; }
  template <class T>
  T* at(std::size_t offset = 0) const {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(data_) + offset);
  }

 private:
  void* data_{};
  std::size_t bytes_{};
};

// Exact-shape row-major BF16 GEMM, FP32 accumulation, BF16 output.
class Linear {
 public:
  Linear(std::uint32_t m, std::uint32_t k, std::uint32_t n) {
    try {
      check(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "MTP GEMM descriptor");
      const cublasOperation_t trans = CUBLAS_OP_T;
      check(cublasLtMatmulDescSetAttribute(
                op_, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)),
            "MTP GEMM transpose");
      layout(&a_, m, k);
      layout(&b_, n, k);
      layout(&c_, m, n);
    } catch (...) {
      release();
      throw;
    }
  }
  ~Linear() { release(); }
  Linear(const Linear&) = delete;
  Linear& operator=(const Linear&) = delete;
  void run(cublasLtHandle_t handle, const __nv_bfloat16* input,
           const __nv_bfloat16* weight, __nv_bfloat16* output,
           cudaStream_t stream) const {
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(handle, op_, &alpha, input, a_, weight, b_, &beta,
                         output, c_, output, c_, nullptr, nullptr, 0, stream),
          "MTP GEMM");
  }

 private:
  static void layout(cublasLtMatrixLayout_t* out, std::uint32_t rows,
                     std::uint32_t cols) {
    check(cublasLtMatrixLayoutCreate(out, CUDA_R_16BF, rows, cols, cols),
          "MTP GEMM layout");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(
              *out, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
          "MTP GEMM row order");
  }

  void release() {
    if (a_) cublasLtMatrixLayoutDestroy(a_);
    if (b_) cublasLtMatrixLayoutDestroy(b_);
    if (c_) cublasLtMatrixLayoutDestroy(c_);
    if (op_) cublasLtMatmulDescDestroy(op_);
    a_ = b_ = c_ = nullptr;
    op_ = nullptr;
  }

  cublasLtMatmulDesc_t op_{};
  cublasLtMatrixLayout_t a_{}, b_{}, c_{};
};

}  // namespace gewell::mtp_cuda
