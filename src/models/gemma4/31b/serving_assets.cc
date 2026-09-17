#include "gewell/models/gemma4/31b/serving_assets.h"
#include "gewell/models/gemma4/text_contract.h"

#include "json.hpp"

#include <openssl/evp.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace gewell::gemma4_31b {
namespace {

using nlohmann::json;
namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("serving assets: " + message);
}

void regular_file(const fs::path& path) {
  std::error_code error;
  const auto status = fs::symlink_status(path, error);
  require(!error && fs::is_regular_file(status),
          "expected a local regular file (no symlinks): " + path.string());
}

std::string read_file(const fs::path& path, std::uintmax_t limit) {
  regular_file(path);
  const auto size = fs::file_size(path);
  require(size <= limit, "file is too large: " + path.string());
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open " + path.string());
  std::string bytes(static_cast<std::size_t>(size), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(size));
  require(input.good() && input.peek() == std::char_traits<char>::eof(),
          "file changed or could not be read: " + path.string());
  return bytes;
}

std::string sha256(const std::string& bytes) {
  artifact::Digest digest;
  unsigned int size = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(), nullptr) == 1 &&
              size == digest.size(), "SHA-256 failed");
  return artifact::digest_hex(digest);
}

void keys(const json& value, const std::initializer_list<const char*> expected,
          const std::string& label) {
  require(value.is_object() && value.size() == expected.size(), label + " has unexpected fields");
  for (const char* key : expected) require(value.contains(key), label + " is missing " + key);
}

}  // namespace

ServingAssets ServingAssets::Open(const std::string& model_directory) {
  const fs::path directory(model_directory);
  const auto& contract = gemma4::text_contract_31b();
  try {
    const auto manifest = json::parse(read_file(directory / "manifest.json", 4 * 1024 * 1024));
    keys(manifest, {"aliases", "artifact", "format", "layout", "model",
                    "schema_version", "serving", "source", "tensors"}, "manifest");
    require(manifest.at("schema_version").is_number_unsigned() && manifest.at("schema_version") == 6,
            "manifest schema must be integer 6; convert the model first");
    const bool native_nvfp4 = manifest.at("format") == "gemma4-31b-nvfp4-w4a4-v2";
    const bool native_mixed = manifest.at("format") == "gemma4-31b-mixed-v2";
    require(native_nvfp4 || native_mixed || manifest.at("format") == "gemma4-31b-bf16-v4", "wrong artifact format");
    const auto& serving = manifest.at("serving");
    keys(serving, {"repository", "revision", "assets"}, "serving");
    require(serving.at("repository").is_string() && serving.at("revision").is_string(),
            "serving provenance must be strings");
    const auto& assets = serving.at("assets");
    require(assets.is_array() && assets.size() == 1, "wrong serving asset inventory");
    const auto& tokenizer = assets[0];
    keys(tokenizer, {"file", "byte_length", "sha256"}, "asset entry");
    require(tokenizer.at("file") == "tokenizer.json", "wrong asset filename for tokenizer.json");
    auto tokenizer_data = read_file(directory / "tokenizer.json", 128 * 1024 * 1024);
    require(tokenizer.at("byte_length").is_number_unsigned() &&
                tokenizer.at("byte_length").get<std::uint64_t>() == tokenizer_data.size(),
            "wrong byte length for tokenizer.json");
    require(tokenizer.at("sha256").is_string() && sha256(tokenizer_data) == tokenizer.at("sha256"),
            "SHA-256 mismatch for tokenizer.json");
    const auto& metadata = manifest.at("artifact");
    const auto filename = metadata.at("file").get<std::string>();
    require(!filename.empty() && fs::path(filename).filename() == filename &&
                filename != "." && filename != "..", "artifact filename must be a local basename");
    regular_file(directory / filename);
    auto weights = artifact::ArtifactFile::Open((directory / filename).string());
    const auto& header = weights.header();
    require(header.native_nvfp4 == native_nvfp4 && header.native_mixed == native_mixed,
            "manifest format does not match artifact");
    require(metadata.at("file_bytes").is_number_unsigned() &&
                metadata.at("file_bytes") == header.file_bytes &&
                metadata.at("header_sha256") == artifact::digest_hex(header.header_sha256) &&
                metadata.at("entry_table_sha256") == artifact::digest_hex(header.entry_table_sha256) &&
                metadata.at("payload_sha256") == artifact::digest_hex(header.payload_sha256),
            "weight manifest does not match artifact metadata");
    require(manifest.at("model").at("config_sha256") == artifact::digest_hex(header.config_sha256) &&
                manifest.at("model").at("index_sha256") == artifact::digest_hex(header.source_index_sha256),
            "model manifest does not match artifact metadata");
    return {std::move(weights), text::Tokenizer::FromJson(tokenizer_data, contract)};
  } catch (const json::exception& error) {
    throw std::runtime_error("serving assets: invalid manifest in " + model_directory + ": " + error.what());
  } catch (const fs::filesystem_error& error) {
    throw std::runtime_error("serving assets: " + std::string(error.what()));
  }
}

}  // namespace gewell::gemma4_31b
