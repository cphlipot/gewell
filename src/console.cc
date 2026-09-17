#include "gewell/console.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace gewell::console {
namespace {
std::atomic<bool> json_format{false};
std::mutex output_mutex;

std::string decimal(double value, unsigned precision = 2) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  auto text = out.str();
  if (text.find('.') != std::string::npos) {
    while (text.back() == '0') text.pop_back();
    if (text.back() == '.') text.pop_back();
  }
  return text;
}

std::string clean(std::string_view text, bool multiline = false) {
  // Use the JSON serializer's UTF-8 replacement policy in both console formats.
  const auto normalized = nlohmann::json::parse(nlohmann::json(text).dump(
      -1, ' ', false, nlohmann::json::error_handler_t::replace)).get<std::string>();
  std::string result;
  for (const unsigned char ch : normalized) {
    if (ch == '\n') result += multiline ? "\n" : "\\n";
    else if (ch == '\r') result += "\\r";
    else if (ch == '\t') result += "\\t";
    else if (ch < 32 || ch == 127) {
      constexpr char hex[] = "0123456789abcdef";
      result += "\\x";
      result += hex[ch >> 4];
      result += hex[ch & 15];
    } else result += static_cast<char>(ch);
  }
  return result;
}

void emit(std::string text, bool error = false) {
  std::lock_guard<std::mutex> lock(output_mutex);
  auto& out = error ? std::cerr : std::cout;
  out << text;
  if (text.empty() || text.back() != '\n') out << '\n';
  out << std::flush;
}

void record(const nlohmann::json& value, bool error = false) {
  emit(value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), error);
}

bool ends_with(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string label(std::string_view name) {
  if (name == "server_batch_capacity") return "Concurrent requests";
  for (const std::string_view prefix : {"server_batch_", "server_", "generation_", "caption_", "batch_", "replay_"}) {
    if (name.substr(0, prefix.size()) == prefix) {
      name.remove_prefix(prefix.size());
      break;
    }
  }
  std::string text(name);
  std::replace(text.begin(), text.end(), '_', ' ');
  std::istringstream words(text);
  std::ostringstream out;
  std::string word;
  bool first = true;
  while (words >> word) {
    if (word == "gpu" || word == "cpu" || word == "kv" || word == "mtp" ||
        word == "cuda" || word == "qdq" || word == "http" || word == "bos" ||
        word == "ram" || word == "vram" || word == "id" || word == "ids") {
      for (auto& ch : word) if (ch >= 'a' && ch <= 'z') ch -= 'a' - 'A';
    } else if (first && word[0] >= 'a' && word[0] <= 'z') word[0] -= 'a' - 'A';
    if (!first) out << ' ';
    out << word;
    first = false;
  }
  return clean(out.str());
}
}  // namespace

void set_format(std::string_view format) {
  if (format != "human" && format != "json")
    throw std::invalid_argument("--log-format must be human or json");
  json_format = format == "json";
}

bool json_enabled() { return json_format.load(); }

std::string number(std::uint64_t value) {
  auto text = std::to_string(value);
  for (std::size_t at = text.size(); at > 3;) {
    at -= 3;
    text.insert(at, 1, ',');
  }
  return text;
}

std::string bytes(std::uint64_t value) {
  constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
  double scaled = static_cast<double>(value);
  unsigned unit = 0;
  while (scaled >= 1024 && unit < 6) { scaled /= 1024; ++unit; }
  return decimal(scaled) + ' ' + units[unit];
}

std::string seconds(double value) {
  if (!std::isfinite(value)) return "n/a";
  if (value != 0 && std::abs(value) < 0.001) return decimal(value * 1e6) + " us";
  if (value != 0 && std::abs(value) < 1) return decimal(value * 1e3) + " ms";
  return decimal(value) + " s";
}

std::string token_rate(std::uint64_t tokens, double gpu_seconds) {
  if (!tokens || !(gpu_seconds > 0) || !std::isfinite(gpu_seconds)) return "n/a";
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << tokens / gpu_seconds << " t/s";
  return out.str();
}

void section(std::string_view title) {
  if (json_enabled()) record({{"event", "section"}, {"title", title}});
  else emit("\n" + clean(title));
}

void field(std::string_view name, const nlohmann::json& value) {
  if (json_enabled()) {
    record({{"event", "field"}, {"name", name}, {"value", value}});
    return;
  }
  std::string_view key = name, qualifier;
  for (const std::string_view suffix : {"_raw", "_median", "_mean", "_min", "_max"}) {
    if (ends_with(key, suffix)) {
      qualifier = key.substr(key.size() - suffix.size());
      key.remove_suffix(suffix.size());
      break;
    }
  }
  std::string_view unit;
  if (value.is_number() || value.is_array()) {
    for (const std::string_view suffix : {"_bytes", "_mib", "_milliseconds", "_seconds", "_tokens_per_second"}) {
      if (ends_with(key, suffix)) {
        unit = suffix;
        key.remove_suffix(suffix.size());
        break;
      }
    }
  }
  const auto show = [&](const nlohmann::json& item) -> std::string {
    if (item.is_number()) {
      const double numeric = item.get<double>();
      if (unit == "_bytes" && numeric >= 0 && item.is_number_integer())
        return bytes(item.get<std::uint64_t>());
      if (unit == "_bytes" && numeric < 0 && item.is_number_integer())
        return "-" + bytes(std::uint64_t(-(item.get<std::int64_t>() + 1)) + 1);
      if (unit == "_mib" && numeric >= 0 &&
          static_cast<long double>(numeric) * 1048576 <= std::numeric_limits<std::uint64_t>::max())
        return bytes(static_cast<std::uint64_t>(numeric * 1048576));
      if (unit == "_milliseconds") return seconds(numeric / 1000);
      if (unit == "_seconds") return seconds(numeric);
      if (unit == "_tokens_per_second") return decimal(numeric) + " tokens/s";
      if (item.is_number_unsigned() || (item.is_number_integer() && numeric >= 0))
        return number(item.get<std::uint64_t>());
      if (item.is_number_integer())
        return "-" + number(std::uint64_t(-(item.get<std::int64_t>() + 1)) + 1);
      std::ostringstream out;
      out << std::setprecision(6) << numeric;
      return out.str();
    }
    if (item.is_string()) return clean(item.get_ref<const std::string&>());
    if (item.is_boolean()) return item.get<bool>() ? "yes" : "no";
    return clean(item.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
  };
  std::string shown;
  if (value.is_array()) {
    shown = "[";
    for (const auto& item : value) {
      if (shown.size() > 1) shown += ", ";
      shown += show(item);
    }
    shown += ']';
  } else shown = show(value);
  std::ostringstream out;
  out << "  " << std::left << std::setw(42) << label(std::string(key) + std::string(qualifier)) << "  " << shown;
  emit(out.str());
}

void event(std::string_view name, const nlohmann::json& data,
           std::string_view human, bool error) {
  if (json_enabled()) record({{"event", name}, {"data", data}}, error);
  else if (!human.empty()) emit(clean(human), error);
}

void message(std::string_view text, bool error) {
  if (json_enabled()) record({{"event", "message"}, {"message", text}}, error);
  else emit((error ? "Error: " : "") + clean(text, !error), error);
}

}  // namespace gewell::console
