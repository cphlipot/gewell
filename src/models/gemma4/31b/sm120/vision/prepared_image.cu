#include "prepared_image.h"

namespace gewell::gemma4_31b::sm120 {
PreparedImage::PreparedImage(const WeightArena& weights, const std::vector<std::uint8_t>& pixels,
    const std::vector<std::uint8_t>& positions, std::uint32_t padded_patch_rows,
    std::uint32_t soft_token_count)
    : features_(std::size_t(soft_token_count) * model::kHiddenSize * sizeof(BFloat16)) {
    DeviceAllocation device_pixels(pixels.size());
    DeviceAllocation device_positions(positions.size());
    check_cuda(cudaMemcpy(device_pixels.data(), pixels.data(), pixels.size(),
                          cudaMemcpyHostToDevice),
               "copy prepared pixels to device");
    check_cuda(cudaMemcpy(device_positions.data(), positions.data(),
                          positions.size(), cudaMemcpyHostToDevice),
               "copy prepared positions to device");

    vision_executor::VisionExecutor tower(
        {weights.pointer(model::kVisionPatchProjectionPhysicalId),
         vision_executor::kVisionWeightSliceBytes},
        soft_token_count);
    scratch_bytes_ = tower.scratch_bytes();
    DeviceAllocation vision_scratch(scratch_bytes_);
    const vision_engine::PrefillRequest request{
        {static_cast<const float*>(device_pixels.data()),
         static_cast<const std::int32_t*>(device_positions.data()),
         padded_patch_rows, soft_token_count},
        features_.data(), soft_token_count};
    CudaEvent begin;
    CudaEvent end;
    check_cuda(cudaEventRecord(begin.get()), "record prepared vision start");
    tower.run(request, vision_scratch.data(), vision_scratch.size(), nullptr,
              nullptr);
    check_cuda(cudaEventRecord(end.get()), "record prepared vision end");
    check_cuda(cudaEventSynchronize(end.get()),
               "synchronize prepared vision execution");
    check_cuda(cudaEventElapsedTime(&gpu_milliseconds_, begin.get(),
                                    end.get()),
               "measure prepared vision execution");
}
}
