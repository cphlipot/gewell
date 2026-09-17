#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include "json.hpp"

namespace gewell::console {

// Console presentation only; persisted results and wire protocols use their own serializers.
void set_format(std::string_view format);
bool json_enabled();
std::string number(std::uint64_t value);
std::string bytes(std::uint64_t value);
std::string seconds(double value);
std::string token_rate(std::uint64_t tokens, double gpu_seconds);
void section(std::string_view title);
void field(std::string_view name, const nlohmann::json& value);
// Events with no human summary are detailed diagnostics, emitted only in JSON mode.
void event(std::string_view name, const nlohmann::json& data,
           std::string_view human = {}, bool error = false);
void message(std::string_view text, bool error = false);

}  // namespace gewell::console
