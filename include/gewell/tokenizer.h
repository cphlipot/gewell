#pragma once

#include "gewell/text_contract.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gewell::text {

// BPE mechanics for the model-validated JSON pipeline. Immutable after construction; encode
// scratch belongs to the caller, so concurrent requests can share this object.
class Tokenizer {
 public:
  Tokenizer(const std::string& tokenizer_json_path, const TextContract& contract);
  static Tokenizer FromJson(std::string_view bytes, const TextContract& contract);
  [[nodiscard]] const TextContract& contract() const { return *contract_; }

  [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const;
  [[nodiscard]] std::vector<std::uint32_t> completion_prompt(std::string_view text) const;
  [[nodiscard]] std::vector<std::uint32_t> completion_prompt(const std::vector<std::uint32_t>& tokens) const;
  [[nodiscard]] const std::string& token_piece(std::uint32_t id) const;
  [[nodiscard]] bool is_special(std::uint32_t id) const;

 private:
  explicit Tokenizer(const TextContract& contract) : contract_(&contract) {}
  void load(std::string_view bytes);
  struct Merge { std::uint32_t rank, token; };
  void encode_span(std::string_view text, std::vector<std::uint32_t>& output) const;

  const TextContract* contract_;
  std::vector<std::string> pieces_;
  std::unordered_map<std::string, std::uint32_t> vocabulary_;
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::vector<std::uint32_t> specials_;
  std::array<std::uint32_t, 256> byte_tokens_{};
};

}  // namespace gewell::text
