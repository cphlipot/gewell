#include "gewell/runtime/image.h"

#include <openssl/evp.h>

#include <memory>
#include <stdexcept>

namespace gewell::runtime {
std::array<std::uint8_t, 32> image_digest(const ImageInput& image) {
  if (image.begin >= image.end)
    throw std::invalid_argument("image identity requires a nonempty feature span");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("cannot initialize image identity SHA-256");
  const auto append = [&](const void* bytes, std::size_t size) {
    if (size && EVP_DigestUpdate(context.get(), bytes, size) != 1)
      throw std::runtime_error("cannot update image identity SHA-256");
  };
  const auto integer = [&](std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (auto& byte : bytes) { byte = value & 0xff; value >>= 8; }
    append(bytes.data(), bytes.size());
  };
  constexpr char domain[] = "gewell.prepared-image.v1";
  append(domain, sizeof(domain) - 1);
  integer(image.padded_patch_rows);
  integer(image.end - image.begin);
  integer(image.pixels.size());
  append(image.pixels.data(), image.pixels.size());
  integer(image.positions.size());
  append(image.positions.data(), image.positions.size());
  std::array<std::uint8_t, 32> digest{};
  unsigned size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != digest.size())
    throw std::runtime_error("cannot finish image identity SHA-256");
  return digest;
}
}  // namespace gewell::runtime
