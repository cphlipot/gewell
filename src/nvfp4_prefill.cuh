#pragma once

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <stdexcept>
#include <string>

namespace gewell::nvfp4::detail {

// Offline selection for the SM120 MLP shapes. The larger tile beats the
// cuBLASLt 128x128 kernel on prefill; small rows retain the cuBLASLt plan.
class PrefillGemm {
  using Tile = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using Output = cutlass::bfloat16_t;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::TmaWarpSpecialized,
      cutlass::epilogue::fusion::LinearCombination<Output, float, void, float>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      Element, cutlass::layout::RowMajor, 32,
      Element, cutlass::layout::ColumnMajor, 32, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Epilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

 public:
  static bool selected(unsigned rows, unsigned k, unsigned n) {
    return rows >= 512 && rows <= 4096 &&
        ((k == 5376 && n == 43008) || (k == 21504 && n == 5376));
  }

  PrefillGemm(int rows, int k, int n) {
    args_.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args_.problem_shape = cute::make_shape(rows, n, k, 1);
    args_.mainloop.dA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {rows, k, 1});
    args_.mainloop.dB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
    args_.epilogue.dC = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {rows, n, 1});
    args_.epilogue.dD = args_.epilogue.dC;
    using Scales = typename Mainloop::Sm1xxBlkScaledConfig;
    args_.mainloop.layout_SFA = Scales::tile_atom_to_shape_SFA(args_.problem_shape);
    args_.mainloop.layout_SFB = Scales::tile_atom_to_shape_SFB(args_.problem_shape);
    args_.scheduler.max_swizzle_size = 1;
    args_.hw_info.cluster_shape = dim3(1, 1, 1);
    args_.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    auto status = cudaGetDevice(&args_.hw_info.device_id);
    if (status == cudaSuccess)
      status = cudaDeviceGetAttribute(&args_.hw_info.sm_count, cudaDevAttrMultiProcessorCount,
                                     args_.hw_info.device_id);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }

  std::size_t workspace_bytes() const { return Gemm::get_workspace_size(args_); }

  void run(const std::uint8_t* input, const std::uint8_t* weight,
           const std::uint8_t* input_scales, const std::uint8_t* weight_scales,
           float alpha, __nv_bfloat16* output, void* workspace, cudaStream_t stream) {
    args_.mainloop.ptr_A = reinterpret_cast<const cutlass::float_e2m1_t*>(input);
    args_.mainloop.ptr_B = reinterpret_cast<const cutlass::float_e2m1_t*>(weight);
    args_.mainloop.ptr_SFA = reinterpret_cast<const cutlass::float_ue4m3_t*>(input_scales);
    args_.mainloop.ptr_SFB = reinterpret_cast<const cutlass::float_ue4m3_t*>(weight_scales);
    args_.epilogue.ptr_D = reinterpret_cast<Output*>(output);
    args_.epilogue.thread.alpha = alpha;
    check(gemm_.initialize(args_, workspace, stream));
    check(gemm_.run(stream));
  }

 private:
  static void check(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("NVFP4 prefill GEMM: ") + cutlass::cutlassGetStatusString(status));
  }
  typename Gemm::Arguments args_{};
  Gemm gemm_;
};

}  // namespace gewell::nvfp4::detail
