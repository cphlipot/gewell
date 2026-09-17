#include "gewell/runtime/image.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void identity_cases() {
  using namespace gewell::runtime;
  ImageInput image{{1, 2, 3}, {4, 5}, 9, 10, 13};
  const auto expected = image_digest(image);
  std::ostringstream hex;
  for (auto byte : expected) hex << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
  require(hex.str() == "c76afdfc5da153ca080f8dbfcc457818dd2fc66a4ade5cdf52e56e01edebd83e",
          "canonical prepared-image digest differs from independent SHA-256 fixture");
  auto shifted = image;
  shifted.begin += 100; shifted.end += 100;
  require(image_digest(shifted) == expected, "absolute position changed tensor identity");
  for (unsigned change = 0; change < 5; ++change) {
    auto changed = image;
    if (change == 0) changed.pixels[1] ^= 1;
    if (change == 1) changed.positions[1] ^= 1;
    if (change == 2) ++changed.padded_patch_rows;
    if (change == 3) ++changed.end;
    if (change == 4) {
      changed.pixels.push_back(changed.positions.front());
      changed.positions.erase(changed.positions.begin());
    }
    require(image_digest(changed) != expected, "prepared tensor bytes or metadata were omitted from identity");
  }
  image.end = image.begin;
  bool rejected = false;
  try { (void)image_digest(image); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "empty feature span was assigned an image identity");
}
}  // namespace

int main() {
  try {
    identity_cases();
    std::cout << "image identity tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
