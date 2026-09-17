#include "gewell/models/gemma4/31b/artifact.h"
#include "../src/models/gemma4/31b/artifact_detail.h"

#include <iostream>
#include <string_view>

namespace {
using namespace gewell::artifact;
using namespace gewell::artifact::detail;

bool run_self_tests(std::string* failure) {
  try {
    const auto expect_hash = [](std::string_view input,
                                std::string_view expected) {
      const auto* bytes =
          reinterpret_cast<const std::uint8_t*>(input.data());
      require(digest_hex(hash_bytes(bytes, input.size())) == expected,
              "SHA-256 vector failed");
    };
    expect_hash("",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect_hash("abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect_hash(
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    Sha256 chunked;
    for (const char value : std::string_view("abc")) {
      chunked.Update(reinterpret_cast<const std::uint8_t*>(&value), 1);
    }
    require(digest_hex(chunked.Finish()) ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "chunked SHA-256 vector failed");

    const std::array<std::uint8_t, 16> encoded{{
        0x34, 0x12, 0xfe, 0xff, 0x78, 0x56, 0x34, 0x12,
        0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
    }};
    require(read_u16_le(encoded.data()) == 0x1234U,
            "u16 little-endian decode failed");
    require(read_i16_le(encoded.data() + 2) == -2,
            "i16 little-endian decode failed");
    require(read_u32_le(encoded.data() + 4) == 0x12345678U,
            "u32 little-endian decode failed");
    require(read_u64_le(encoded.data() + 8) == 0x0123456789abcdefULL,
            "u64 little-endian decode failed");
    require(checked_align_up(1, 4'096, "self-test") == 4'096,
            "alignment invariant failed");
    require(checked_align_up(4'096, 4'096, "self-test") == 4'096,
            "aligned-value invariant failed");
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}  // namespace

int main() {
  std::string failure;
  if (!run_self_tests(&failure)) {
    std::cerr << failure << "\n";
    return 1;
  }
  std::cout << "artifact tests: ok\n";
  return 0;
}
