#pragma once

#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/tokenizer.h"

#include <string>

namespace gewell::gemma4_31b {

// Opens only local files. Weight metadata uses ArtifactFile's existing checks;
// tokenizer.json is hash-verified against the bundle manifest.
struct ServingAssets {
  artifact::ArtifactFile weights;
  text::Tokenizer tokenizer;

  static ServingAssets Open(const std::string& model_directory);
};

}  // namespace gewell::gemma4_31b
