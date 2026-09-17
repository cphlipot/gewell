#include "bf16_common.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLASLt status " + std::to_string(status));
  }
}

class MatmulDescription {
 public:
  MatmulDescription() {
    check_cublas(cublasLtMatmulDescCreate(&value_, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F),
                 "cublasLtMatmulDescCreate");
  }

  ~MatmulDescription() {
    if (value_ != nullptr) {
      cublasLtMatmulDescDestroy(value_);
    }
  }

  MatmulDescription(const MatmulDescription&) = delete;
  MatmulDescription& operator=(const MatmulDescription&) = delete;

  cublasLtMatmulDesc_t get() const { return value_; }

 private:
  cublasLtMatmulDesc_t value_{nullptr};
};

class MatrixLayout {
 public:
  MatrixLayout(std::uint64_t rows, std::uint64_t columns,
               std::int64_t leading_dimension) {
    check_cublas(cublasLtMatrixLayoutCreate(&value_, CUDA_R_16BF, rows,
                                            columns, leading_dimension),
                 "cublasLtMatrixLayoutCreate");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(
                     value_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                     sizeof(order)),
                 "set row-major matrix layout");
  }

  ~MatrixLayout() {
    if (value_ != nullptr) {
      cublasLtMatrixLayoutDestroy(value_);
    }
  }

  MatrixLayout(const MatrixLayout&) = delete;
  MatrixLayout& operator=(const MatrixLayout&) = delete;

  cublasLtMatrixLayout_t get() const { return value_; }

 private:
  cublasLtMatrixLayout_t value_{nullptr};
};

}  // namespace

void linear_m1(cublasLtHandle_t handle, const BFloat16* input,
               const BFloat16* weight, BFloat16* output,
               std::uint32_t input_width, std::uint32_t output_width,
               cudaStream_t stream) {
  check_pointer(handle, "linear_m1 handle");
  check_pointer(input, "linear_m1 input");
  check_pointer(weight, "linear_m1 weight");
  check_pointer(output, "linear_m1 output");
  if (input_width == 0 || output_width == 0) {
    fail("linear_m1", "matrix dimensions must be positive");
  }

  MatmulDescription operation;
  const cublasOperation_t transpose_a = CUBLAS_OP_N;
  const cublasOperation_t transpose_b = CUBLAS_OP_T;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   operation.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
                   sizeof(transpose_a)),
               "set linear_m1 TRANSA");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   operation.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
                   sizeof(transpose_b)),
               "set linear_m1 TRANSB");

  MatrixLayout input_layout(1, input_width, input_width);
  MatrixLayout weight_layout(output_width, input_width, input_width);
  MatrixLayout output_layout(1, output_width, output_width);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  check_cublas(
      cublasLtMatmul(handle, operation.get(), &alpha, input,
                     input_layout.get(), weight, weight_layout.get(), &beta,
                     output, output_layout.get(), output, output_layout.get(),
                     nullptr, nullptr, 0, stream),
      "linear_m1 cublasLtMatmul");
}

}  // namespace gewell::bf16_primitives
