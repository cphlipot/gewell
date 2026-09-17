#pragma once

#include "gewell/models/gemma4/31b/artifact.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>

namespace gewell::artifact::detail {

[[noreturn]] inline void fail(const std::string& message) { throw Error(message); }

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

inline std::uint16_t read_u16_le(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

inline std::int16_t read_i16_le(const std::uint8_t* bytes) {
  const std::uint16_t value = read_u16_le(bytes);
  const std::int32_t signed_value =
      value <= 0x7fffU ? static_cast<std::int32_t>(value)
                       : static_cast<std::int32_t>(value) - 0x1'0000;
  return static_cast<std::int16_t>(signed_value);
}

inline std::uint32_t read_u32_le(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t read_u64_le(const std::uint8_t* bytes) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
  }
  return value;
}

inline std::uint32_t rotate_right(std::uint32_t value, unsigned count) {
  return (value >> count) | (value << (32 - count));
}

class Sha256 {
 public:
  void Update(const std::uint8_t* bytes, std::size_t length) {
    if (length > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
      fail("SHA-256 input length overflow");
    }
    total_bytes_ += length;

    if (buffered_ != 0) {
      const std::size_t taken = std::min(length, buffer_.size() - buffered_);
      if (taken != 0) {
        std::memcpy(buffer_.data() + buffered_, bytes, taken);
        buffered_ += taken;
        bytes += taken;
        length -= taken;
      }
      if (buffered_ == buffer_.size()) {
        Transform(buffer_.data());
        buffered_ = 0;
      }
    }

    while (length >= buffer_.size()) {
      Transform(bytes);
      bytes += buffer_.size();
      length -= buffer_.size();
    }
    if (length != 0) {
      std::memcpy(buffer_.data(), bytes, length);
      buffered_ = length;
    }
  }

  [[nodiscard]] Digest Finish() {
    require(total_bytes_ <= std::numeric_limits<std::uint64_t>::max() / 8,
            "SHA-256 bit length overflow");
    const std::uint64_t bit_length = total_bytes_ * 8;

    buffer_[buffered_++] = 0x80;
    if (buffered_ > 56) {
      std::fill(buffer_.begin() + buffered_, buffer_.end(), 0);
      Transform(buffer_.data());
      buffered_ = 0;
    }
    std::fill(buffer_.begin() + buffered_, buffer_.begin() + 56, 0);
    for (std::size_t index = 0; index < 8; ++index) {
      buffer_[63 - index] =
          static_cast<std::uint8_t>(bit_length >> (index * 8));
    }
    Transform(buffer_.data());
    buffered_ = 0;

    Digest digest{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      digest[index * 4] = static_cast<std::uint8_t>(state_[index] >> 24);
      digest[index * 4 + 1] = static_cast<std::uint8_t>(state_[index] >> 16);
      digest[index * 4 + 2] = static_cast<std::uint8_t>(state_[index] >> 8);
      digest[index * 4 + 3] = static_cast<std::uint8_t>(state_[index]);
    }
    return digest;
  }

 private:
  void Transform(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> constants{{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    }};

    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::uint8_t* input = block + index * 4;
      words[index] = (static_cast<std::uint32_t>(input[0]) << 24) |
                     (static_cast<std::uint32_t>(input[1]) << 16) |
                     (static_cast<std::uint32_t>(input[2]) << 8) |
                     static_cast<std::uint32_t>(input[3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t first =
          rotate_right(words[index - 15], 7) ^
          rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const std::uint32_t second =
          rotate_right(words[index - 2], 17) ^
          rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + first + words[index - 7] + second;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                 rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t first =
          h + sum1 + choose + constants[index] + words[index];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                 rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t second = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{{
      0x6a09e667U,
      0xbb67ae85U,
      0x3c6ef372U,
      0xa54ff53aU,
      0x510e527fU,
      0x9b05688cU,
      0x1f83d9abU,
      0x5be0cd19U,
  }};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_{};
  std::uint64_t total_bytes_{};
};

inline Digest hash_bytes(const std::uint8_t* bytes, std::size_t length) {
  Sha256 digest;
  digest.Update(bytes, length);
  return digest.Finish();
}

inline std::uint64_t checked_align_up(std::uint64_t value, std::uint64_t alignment,
                               const std::string& label) {
  require(alignment != 0, label + " has zero alignment");
  require(value <= std::numeric_limits<std::uint64_t>::max() - (alignment - 1),
          label + " alignment overflows u64");
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace gewell::artifact::detail
