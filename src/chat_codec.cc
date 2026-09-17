#include "gewell/chat_codec.h"
#include "text/utf8.h"

namespace gewell::text {
namespace {

using detail::read_scalar;
constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
int hex_digit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

std::string replace_space_markers(std::string_view piece, std::string_view space_marker) {
  std::string result;
  std::size_t offset = 0;
  for (;;) {
    const auto found = piece.find(space_marker, offset);
    result.append(piece.substr(offset, found == std::string_view::npos
                                         ? found : found - offset));
    if (found == std::string_view::npos) return result;
    result.push_back(' ');
    offset = found + space_marker.size();
  }
}

}  // namespace

IncrementalTextDecoder::IncrementalTextDecoder(const Tokenizer& tokenizer,
                                               bool skip_special_tokens)
    : tokenizer_(tokenizer), skip_special_tokens_(skip_special_tokens) {}

std::string IncrementalTextDecoder::push(std::uint32_t token) {
  const std::string& piece = tokenizer_.token_piece(token);
  if (skip_special_tokens_ && tokenizer_.is_special(token)) return {};
  if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece.back() == '>') {
    const int high = hex_digit(piece[3]);
    const int low = hex_digit(piece[4]);
    if (high >= 0 && low >= 0) {
      bytes_.push_back(static_cast<char>((high << 4) | low));
      return {};
    }
  }
  return finish() + replace_space_markers(piece, tokenizer_.contract().space_marker);
}

std::string IncrementalTextDecoder::finish() {
  std::size_t offset = 0;
  while (offset < bytes_.size()) {
    std::uint32_t scalar = 0;
    if (!read_scalar(bytes_, offset, scalar)) {
      std::string replacement;
      replacement.reserve(bytes_.size() * kReplacement.size());
      for (std::size_t index = 0; index < bytes_.size(); ++index) {
        replacement += kReplacement;
      }
      bytes_.clear();
      return replacement;
    }
  }
  std::string result;
  result.swap(bytes_);
  return result;
}

std::string IncrementalTextDecoder::pending() const {
  std::size_t offset = 0;
  while (offset < bytes_.size()) {
    std::uint32_t scalar = 0;
    if (!read_scalar(bytes_, offset, scalar)) return {};
  }
  return bytes_;
}

ChatOutputDecoder::ChatOutputDecoder(const Tokenizer& tokenizer)
    : decoder_(tokenizer), contract_(tokenizer.contract()) {}

ChatDelta ChatOutputDecoder::push(std::uint32_t token) {
  if (token == contract_.channel_start || token == contract_.channel_end) {
    ChatDelta delta = finish();
    reasoning_ = token == contract_.channel_start;
    reasoning_prefix_ = reasoning_ ? std::optional<std::string>("") : std::nullopt;
    return delta;
  }
  return route(decoder_.push(token));
}

ChatDelta ChatOutputDecoder::route(std::string text) {
  if (!reasoning_) return {{}, std::move(text)};
  if (reasoning_prefix_) {
    text = *reasoning_prefix_ + text;
    if (contract_.thought_prefix.substr(0, text.size()) == text) {
      reasoning_prefix_ = std::move(text);
      return {};
    }
    reasoning_prefix_.reset();
    if (text.compare(0, contract_.thought_prefix.size(), contract_.thought_prefix) == 0) {
      text.erase(0, contract_.thought_prefix.size());
    }
  }
  return {std::move(text), {}};
}

ChatDelta ChatOutputDecoder::finish() {
  ChatDelta delta = route(decoder_.finish());
  if (reasoning_ && reasoning_prefix_) {
    if (*reasoning_prefix_ != contract_.thought_prefix) delta.reasoning += *reasoning_prefix_;
    reasoning_prefix_.reset();
  }
  return delta;
}

ChatDelta ChatOutputDecoder::pending() const {
  auto copy = *this;
  return copy.route(decoder_.pending());
}

}  // namespace gewell::text
