#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace gewell::text::detail {

// Decode one valid scalar, rejecting overlong forms, surrogates, and scalars
// beyond U+10FFFF. Both trim and ByteFallback need exactly this UTF-8 check.
inline bool read_scalar(std::string_view text, std::size_t& offset,
                 std::uint32_t& scalar) {
  const auto first = static_cast<unsigned char>(text[offset]);
  std::size_t width = 0;
  if (first < 0x80) {
    width = 1;
    scalar = first;
  } else if (first >= 0xC2 && first <= 0xDF) {
    width = 2;
    scalar = first & 0x1F;
  } else if (first >= 0xE0 && first <= 0xEF) {
    width = 3;
    scalar = first & 0x0F;
  } else if (first >= 0xF0 && first <= 0xF4) {
    width = 4;
    scalar = first & 0x07;
  } else {
    return false;
  }
  if (text.size() - offset < width) return false;
  for (std::size_t index = 1; index < width; ++index) {
    const auto byte = static_cast<unsigned char>(text[offset + index]);
    if ((byte & 0xC0) != 0x80) return false;
    scalar = (scalar << 6) | (byte & 0x3F);
  }
  if ((width == 2 && scalar < 0x80) ||
      (width == 3 && scalar < 0x800) ||
      (width == 4 && scalar < 0x10000) || scalar > 0x10FFFF ||
      (scalar >= 0xD800 && scalar <= 0xDFFF)) {
    return false;
  }
  offset += width;
  return true;
}

}  // namespace gewell::text::detail
