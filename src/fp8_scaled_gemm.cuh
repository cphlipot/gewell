#pragma once

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <stdexcept>
#include <string>

namespace gewell::fp8::detail {

// W[M,K] * X[N,K]^T, with one FP32 output scale per M channel and batch.
// Applying scales in the epilogue preserves independent projection scales
// without requantizing weights or materializing an FP32 product matrix.
template<class Output, int TileM = 128, int TileN = 64, int TileK = 128>
class ScaledGemm {
  using Tile = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Element = cutlass::float_e4m3_t;
  using Scale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, Tile, float, float, cute::Stride<cute::_1, cute::_0, int64_t>>;
  using Product = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
          cutlass::FloatRoundStyle::round_to_nearest>,
      Scale, cutlass::epilogue::fusion::Sm90AccFetch>;
  using Fusion = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add,
          Output, float, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,
      cutlass::epilogue::fusion::Sm90SrcFetch<Output>, Product>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      Output, cutlass::layout::ColumnMajor, 16 / sizeof(Output),
      Output, cutlass::layout::ColumnMajor, 16 / sizeof(Output),
      cutlass::epilogue::TmaWarpSpecialized, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 16,
      Element, cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Epilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

 public:
  ScaledGemm(int m, int n, int k, int batches = 1, int ldc = 0,
             std::size_t workspace_limit = 32 * 1024 * 1024) : channels_(m) {
    if (!ldc) ldc = m;
    args_.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args_.problem_shape = cute::make_shape(m, n, k, batches);
    args_.mainloop.dA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, batches});
    args_.mainloop.dB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, batches});
    args_.epilogue.dC = {cute::_1{}, ldc, int64_t(ldc) * n};
    args_.epilogue.dD = args_.epilogue.dC;
    args_.scheduler.max_swizzle_size = 1;
    args_.hw_info.cluster_shape = dim3(1, 1, 1);
    args_.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    auto status = cudaGetDevice(&args_.hw_info.device_id);
    if (status == cudaSuccess)
      status = cudaDeviceGetAttribute(&args_.hw_info.sm_count, cudaDevAttrMultiProcessorCount,
                                     args_.hw_info.device_id);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    // Stream-K can reserve more for reduction than the caller's fixed budget
    // at some odd shapes. The same kernel's data-parallel schedule needs none.
    if (workspace_bytes() > workspace_limit)
      args_.scheduler.decomposition_mode = decltype(args_.scheduler.decomposition_mode)::DataParallel;
  }

  std::size_t workspace_bytes() const { return Gemm::get_workspace_size(args_); }

  void run(const void* weight, const void* input, const float* scales,
           void* output, bool accumulate, void* workspace, cudaStream_t stream) {
    args_.mainloop.ptr_A = static_cast<const Element*>(weight);
    args_.mainloop.ptr_B = static_cast<const Element*>(input);
    args_.epilogue.ptr_C = static_cast<const Output*>(output);
    args_.epilogue.ptr_D = static_cast<Output*>(output);
    args_.epilogue.thread = {
        {{accumulate ? 1.0F : 0.0F}}, {},
        {{scales, 0.0F, {cute::_1{}, cute::_0{}, int64_t(channels_)}}, {}, {}}, {}};
    check(gemm_.initialize(args_, workspace, stream));
    check(gemm_.run(stream));
  }

 private:
  static void check(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("FP8 scaled GEMM: ") + cutlass::cutlassGetStatusString(status));
  }
  int channels_;
  typename Gemm::Arguments args_{};
  Gemm gemm_;
};

}  // namespace gewell::fp8::detail
