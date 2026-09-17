#include "gewell/tokenizer.h"

#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace gewell::text {
namespace {


void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("tokenizer: " + message);
}

std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

// Reject invalid UTF-8 rather than treating arbitrary input bytes as Unicode.
std::size_t character_bytes(std::string_view text, std::size_t at) {
  const auto first = static_cast<unsigned char>(text[at]);
  if (first < 0x80) return 1;
  const std::size_t size = first >= 0xc2 && first <= 0xdf ? 2 :
      first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
  require(size && size <= text.size() - at, "input is not valid UTF-8");
  for (std::size_t i = 1; i < size; ++i)
    require((static_cast<unsigned char>(text[at + i]) & 0xc0) == 0x80,
            "input is not valid UTF-8");
  const auto second = static_cast<unsigned char>(text[at + 1]);
  require(!(first == 0xe0 && second < 0xa0) && !(first == 0xed && second >= 0xa0) &&
              !(first == 0xf0 && second < 0x90) && !(first == 0xf4 && second >= 0x90),
          "input is not valid UTF-8");
  return size;
}

std::uint32_t token_id(const nlohmann::json& value, std::uint32_t vocabulary_size) {
  require(value.is_number_unsigned(), "vocabulary IDs must be unsigned integers");
  const auto id = value.get<std::uint64_t>();
  require(id < vocabulary_size, "vocabulary ID is out of range");
  return static_cast<std::uint32_t>(id);
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path, const TextContract& contract)
    : contract_(&contract) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open " + path);
  const std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  require(!input.bad(), "cannot read " + path);
  load(bytes);
}

Tokenizer Tokenizer::FromJson(std::string_view bytes, const TextContract& contract) {
  Tokenizer tokenizer(contract);
  tokenizer.load(bytes);
  return tokenizer;
}

void Tokenizer::load(std::string_view bytes) {
  nlohmann::json data;
  try {
    data = nlohmann::json::parse(bytes);
    using nlohmann::json;
    contract().validate_tokenizer_pipeline(data);
    const auto& model = data.at("model");
    const auto& vocab = model.at("vocab");
    require(vocab.is_object() && vocab.size() == contract().vocabulary_size, "wrong vocabulary size");
    pieces_.resize(contract().vocabulary_size);
    vocabulary_.reserve(contract().vocabulary_size);
    std::vector<bool> seen(contract().vocabulary_size);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
      const auto id = token_id(it.value(), contract().vocabulary_size);
      require(!seen[id] && !it.key().empty(), "duplicate ID or empty vocabulary token");
      seen[id] = true;
      pieces_[id] = it.key();
      vocabulary_.emplace(it.key(), id);
    }
    const auto& added = data.at("added_tokens");
    require(added.is_array() && added.size() == contract().special_tokens.size(), "wrong added-token inventory");
    for (const auto& entry : added) {
      const auto id = token_id(entry.at("id"), contract().vocabulary_size);
      require(entry.at("content") == pieces_[id] && entry.at("special") == true &&
                  entry.at("normalized") == false && entry.at("single_word") == false &&
                  entry.at("lstrip") == false && entry.at("rstrip") == false,
              "unsupported added-token settings");
      require(std::find(specials_.begin(), specials_.end(), id) == specials_.end(),
              "duplicate added-token ID");
      specials_.push_back(id);
    }
    require(specials_ == contract().special_tokens, "wrong special-token IDs");
    for (const auto& [id, meaning] : contract().control_tokens)
      require(id < pieces_.size() && pieces_[id] == meaning, "wrong control-token meanings");
    constexpr char hex[] = "0123456789ABCDEF";
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
      std::string piece = "<0x00>";
      piece[3] = hex[byte >> 4];
      piece[4] = hex[byte & 15];
      const auto found = vocabulary_.find(piece);
      require(found != vocabulary_.end(), "missing byte-fallback token " + piece);
      byte_tokens_[byte] = found->second;
    }
    const auto& merges = model.at("merges");
    require(merges.is_array() && merges.size() == contract().merge_count, "wrong merge inventory");
    merges_.reserve(contract().merge_count);
    std::uint32_t rank = 0;
    for (const auto& entry : merges) {
      require(entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
                  entry[1].is_string(), "malformed merge");
      const auto& left = entry[0].get_ref<const std::string&>();
      const auto& right = entry[1].get_ref<const std::string&>();
      const auto l = vocabulary_.find(left), r = vocabulary_.find(right);
      const auto joined = vocabulary_.find(left + right);
      require(l != vocabulary_.end() && r != vocabulary_.end() && joined != vocabulary_.end(),
              "merge references a missing token");
      require(merges_.emplace(pair_key(l->second, r->second), Merge{rank++, joined->second}).second,
              "duplicate merge");
    }
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error("tokenizer: invalid JSON: " + std::string(error.what()));
  }
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<std::uint32_t> output;
  std::size_t span = 0, scan = 0;
  while ((scan = text.find('<', scan)) != std::string_view::npos) {
    std::uint32_t special = 0;
    bool found = false;
    // Added tokens match original text before normalization, even inside words.
    // All 24 pinned tokens have no strip/word-boundary options.
    for (const auto id : specials_) {
      if (text.substr(scan, pieces_[id].size()) == pieces_[id] &&
          (!found || pieces_[id].size() > pieces_[special].size())) {
        special = id;
        found = true;
      }
    }
    if (found) {
      encode_span(text.substr(span, scan - span), output);
      output.push_back(special);
      scan += pieces_[special].size();
      span = scan;
    } else ++scan;
  }
  encode_span(text.substr(span), output);
  return output;
}

void Tokenizer::encode_span(std::string_view text, std::vector<std::uint32_t>& output) const {
  if (text.empty()) return;
  constexpr auto absent = std::numeric_limits<std::size_t>::max();
  struct Symbol {
    std::uint32_t token;
    std::size_t previous, next;
    bool live{true};
  };
  std::vector<Symbol> symbols;
  symbols.reserve(text.size());
  auto append = [&](std::uint32_t token) {
    const auto index = symbols.size();
    if (index) symbols.back().next = index;
    symbols.push_back({token, index ? index - 1 : absent, absent});
  };
  for (std::size_t at = 0; at < text.size();) {
    const auto width = character_bytes(text, at);
    const auto original = text.substr(at, width);
    const auto character = original == " " ? contract().space_marker : original;
    const auto found = vocabulary_.find(std::string(character));
    if (found != vocabulary_.end()) append(found->second);
    else for (const unsigned char byte : character) append(byte_tokens_[byte]);
    at += width;
  }
  struct Candidate {
    std::uint32_t rank, token, left_token, right_token;
    std::size_t left, right;
    bool operator<(const Candidate& other) const {
      return rank != other.rank ? rank > other.rank : left > other.left;
    }
  };
  std::priority_queue<Candidate> queue;
  auto offer = [&](std::size_t left) {
    if (left == absent || symbols[left].next == absent) return;
    const auto right = symbols[left].next;
    const auto found = merges_.find(pair_key(symbols[left].token, symbols[right].token));
    if (found != merges_.end()) queue.push({found->second.rank, found->second.token,
        symbols[left].token, symbols[right].token, left, right});
  };
  for (std::size_t index = 0; index + 1 < symbols.size(); ++index) offer(index);
  while (!queue.empty()) {
    const auto item = queue.top();
    queue.pop();
    auto& left = symbols[item.left];
    auto& right = symbols[item.right];
    if (!left.live || !right.live || left.next != item.right ||
        left.token != item.left_token || right.token != item.right_token) continue;
    left.token = item.token;
    left.next = right.next;
    right.live = false;
    if (left.next != absent) symbols[left.next].previous = item.left;
    offer(left.previous);
    offer(item.left);
  }
  for (std::size_t index = 0; index != absent; index = symbols[index].next)
    output.push_back(symbols[index].token);
}

std::vector<std::uint32_t> Tokenizer::completion_prompt(std::string_view text) const {
  auto tokens = encode(text);
  if (tokens.empty() || tokens.front() != contract().bos) tokens.insert(tokens.begin(), contract().bos);
  return tokens;
}

std::vector<std::uint32_t> Tokenizer::completion_prompt(const std::vector<std::uint32_t>& tokens) const {
  for (const auto id : tokens) (void)token_piece(id);
  return tokens;
}

const std::string& Tokenizer::token_piece(std::uint32_t id) const {
  require(id < pieces_.size(), "token ID is outside the vocabulary");
  return pieces_[id];
}

bool Tokenizer::is_special(std::uint32_t id) const {
  require(id < pieces_.size(), "token ID is outside the vocabulary");
  return std::binary_search(specials_.begin(), specials_.end(), id);
}

}  // namespace gewell::text
