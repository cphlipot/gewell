#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace gewell::runtime {
// Native processor output. Positions and pixels keep their model-defined byte
// representation until the selected backend uploads them; [begin,end) is the
// image feature span in the complete tokenized prompt.
struct ImageInput {
  std::vector<std::uint8_t> pixels, positions;
  std::uint32_t padded_patch_rows{}, begin{}, end{};
};

// The prepared tensor representation identifies preprocessing as well as image
// content. Absolute prompt positions are tracked separately by the prefix index.
std::array<std::uint8_t, 32> image_digest(const ImageInput& image);
}  // namespace gewell::runtime
