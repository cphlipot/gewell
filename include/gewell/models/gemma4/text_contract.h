#pragma once

#include "gewell/text_contract.h"

namespace gewell::gemma4 {

// Pinned 31B assets and verified Gemma text semantics. Other variants require
// their own asset/limit contract before sharing these mechanics.
[[nodiscard]] const text::TextContract& text_contract_31b();



}  // namespace gewell::gemma4
