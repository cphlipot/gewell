#pragma once

#include "gewell/models/gemma4/text_contract.h"
#include "gewell/tokenizer.h"

// A small byte-fallback vocabulary exercises the injected contract without
// loading another model or claiming support for another architecture.
inline gewell::text::TextContract small_text_contract() {
  auto contract = gewell::gemma4::text_contract_31b();
  contract.vocabulary_size = 266;
  contract.context_tokens = 16;
  contract.merge_count = 0;
  contract.bos = 256;
  contract.stop_tokens = {257, 258, 263};
  contract.constraint_stop_tokens = {257, 258};
  contract.channel_start = 259;
  contract.channel_end = 260;
  contract.tool_call_start = 261;
  contract.tool_call_end = 262;
  contract.tool_handoff = 263;
  contract.string_delimiter = 264;
  contract.thought_prefix = "plan\n";
  contract.control_tokens = {{256, "<bos>"}, {257, "<eos>"}, {258, "<turn|>"},
      {259, "<|channel>"}, {260, "<channel|>"}, {261, "<|tool_call>"},
      {262, "<tool_call|>"}, {263, "<|tool_response>"}, {264, "<|\"|>"}, {265, "<unk>"}};
  contract.special_tokens.clear();
  for (const auto& [id, meaning] : contract.control_tokens) contract.special_tokens.push_back(id);
  return contract;
}

inline gewell::text::Tokenizer small_tokenizer(const gewell::text::TextContract& contract) {
  using nlohmann::json;
  auto data = json::parse(R"({
    "version":"1.0","truncation":null,"padding":null,
    "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
    "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
    "post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}},
    "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
    "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,"ignore_merges":false,"continuing_subword_prefix":null,"end_of_word_suffix":null,"vocab":{},"merges":[]},
    "added_tokens":[]})");
  constexpr char hex[] = "0123456789ABCDEF";
  for (std::uint32_t byte = 0; byte < 256; ++byte) {
    std::string piece = "<0x00>";
    piece[3] = hex[byte >> 4];
    piece[4] = hex[byte & 15];
    data["model"]["vocab"][piece] = byte;
  }
  for (const auto& [id, meaning] : contract.control_tokens) {
    data["model"]["vocab"][std::string(meaning)] = id;
    data["added_tokens"].push_back({{"id", id}, {"content", meaning}, {"special", true},
        {"normalized", false}, {"single_word", false}, {"lstrip", false}, {"rstrip", false}});
  }
  return gewell::text::Tokenizer::FromJson(data.dump(), contract);
}
