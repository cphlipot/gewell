#include <curl/curl.h>
#include <openssl/evp.h>
#include <json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
double seconds(Time later, Time earlier) {
  return std::chrono::duration<double>(later - earlier).count();
}
Time after(Time start, double delay) {
  return start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(delay));
}
std::string read_file(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  require(bool(file), "cannot open " + path.string());
  std::ostringstream data;
  data << file.rdbuf();
  require(!file.bad(), "cannot read " + path.string());
  return data.str();
}
std::string sha256(const std::string& value) {
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  require(EVP_Digest(value.data(), value.size(), digest, &length, EVP_sha256(), nullptr) == 1,
          "SHA-256 failed");
  std::ostringstream out;
  for (unsigned int i = 0; i < length; ++i)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[i]);
  return out.str();
}
void write_json(const fs::path& path, const Json& value) {
  std::ofstream out(path);
  out << value.dump(2) << '\n';
  out.close();
  require(bool(out), "cannot write " + path.string());
}
void keys(const Json& object, std::initializer_list<const char*> allowed, const std::string& where) {
  require(object.is_object(), where + " must be an object");
  for (const auto& entry : object.items()) {
    require(std::any_of(allowed.begin(), allowed.end(), [&](const char* key) { return entry.key() == key; }),
            where + ": unknown field " + entry.key());
  }
}
std::uint64_t integer(const Json& value, std::uint64_t minimum, std::uint64_t maximum,
                      const std::string& name) {
  require(value.is_number_integer() && !(value.is_number_integer() && !value.is_number_unsigned() &&
          value.get<std::int64_t>() < 0), name + " must be a nonnegative integer");
  const auto result = value.get<std::uint64_t>();
  require(result >= minimum && result <= maximum, name + " is out of range");
  return result;
}
double number(const Json& value, double minimum, double maximum, const std::string& name) {
  require(value.is_number(), name + " must be a number");
  const double result = value.get<double>();
  require(std::isfinite(result) && result >= minimum && result <= maximum, name + " is out of range");
  return result;
}
double cli_number(const std::string& value, double minimum, double maximum, const std::string& name) {
  std::size_t end = 0;
  const double result = std::stod(value, &end);
  require(end == value.size(), "invalid " + name);
  return number(result, minimum, maximum, name);
}

struct Options {
  std::string base_url, model, scenario_file, output, scenario, label, api_key;
  double timeout = 900, connect_timeout = 10;
  std::size_t max_response_bytes = 16 * 1024 * 1024;
  bool plan = false;
};
const char* usage = R"(Usage: gewell-http-bench --base-url URL --model NAME --scenarios FILE --output NEW_DIR
                        [--scenario NAME] [--label LABEL] [--plan]
                        [--api-key-env NAME | --api-key-file FILE]
                        [--timeout-seconds N] [--connect-timeout-seconds N]
                        [--max-response-bytes N]

URL is the API prefix, for example http://127.0.0.1:6311/v1.
--plan validates and prints the run matrix without HTTP or output files.
--help and --version require no other arguments.
)";

Options arguments(int argc, char** argv) {
  Options result;
  bool supplied_key = false;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(seen.insert(key).second, "duplicate option " + key);
    if (key == "--plan") { result.plan = true; continue; }
    require(i + 1 < argc, key + " requires a value");
    const std::string value = argv[++i];
    if (key == "--base-url") result.base_url = value;
    else if (key == "--model") result.model = value;
    else if (key == "--scenarios") result.scenario_file = value;
    else if (key == "--output") result.output = value;
    else if (key == "--scenario") result.scenario = value;
    else if (key == "--label") result.label = value;
    else if (key == "--timeout-seconds") result.timeout = cli_number(value, .001, 86400, key);
    else if (key == "--connect-timeout-seconds") result.connect_timeout = cli_number(value, .001, 86400, key);
    else if (key == "--max-response-bytes") {
      const double size = cli_number(value, 1, 1024.0 * 1024 * 1024, key);
      require(size == std::floor(size), key + " must be an integer");
      result.max_response_bytes = static_cast<std::size_t>(size);
    } else if (key == "--api-key-env" || key == "--api-key-file") {
      require(!supplied_key, "choose one API key source");
      supplied_key = true;
      if (key == "--api-key-env") {
        const char* secret = std::getenv(value.c_str());
        require(secret && *secret, "API key environment variable is unset or empty");
        result.api_key = secret;
      } else {
        result.api_key = read_file(value);
        while (!result.api_key.empty() && (result.api_key.back() == '\n' || result.api_key.back() == '\r'))
          result.api_key.pop_back();
      }
      require(!result.api_key.empty() && std::all_of(result.api_key.begin(), result.api_key.end(),
              [](unsigned char c) { return c >= 33 && c <= 126; }), "API key must contain printable non-whitespace ASCII");
    } else throw std::runtime_error("unknown option " + key);
  }
  require(!result.base_url.empty() && !result.model.empty() && !result.scenario_file.empty(), usage);
  require(result.plan || !result.output.empty(), "--output is required unless --plan is used");
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
  require(bool(url) && curl_url_set(url.get(), CURLUPART_URL, result.base_url.c_str(), 0) == CURLUE_OK,
          "invalid base URL");
  char* part = nullptr;
  require(curl_url_get(url.get(), CURLUPART_SCHEME, &part, 0) == CURLUE_OK, "URL needs a scheme");
  const std::string scheme = part;
  curl_free(part);
  require(scheme == "http" || scheme == "https", "URL scheme must be http or https");
  for (auto field : {CURLUPART_USER, CURLUPART_PASSWORD, CURLUPART_QUERY, CURLUPART_FRAGMENT}) {
    part = nullptr;
    const auto code = curl_url_get(url.get(), field, &part, 0);
    curl_free(part);
    require(code != CURLUE_OK, "base URL must not contain credentials, a query, or a fragment");
  }
  while (!result.base_url.empty() && result.base_url.back() == '/') result.base_url.pop_back();
  return result;
}

struct Case {
  std::string name;
  Json body, followups = Json::array();
  double cancel_after = 0;
  curl_off_t receive_rate = 0;
};
struct Scenario {
  std::string name;
  bool chat = true;
  std::vector<Case> cases;
  std::vector<std::size_t> concurrency;
  std::vector<double> arrival_rates;
  std::size_t sessions = 16, warmups = 2, repetitions = 3;
};

std::vector<Scenario> scenarios(const Json& document, const Options& options) {
  keys(document, {"version", "scenarios"}, "scenario document");
  require(document.contains("version") && integer(document.at("version"), 1, 1, "version") == 1,
          "scenario version must be 1");
  const auto& list = document.at("scenarios");
  require(list.is_array() && !list.empty(), "scenarios must be a nonempty array");
  std::set<std::string> names;
  std::vector<Scenario> result;
  for (const auto& item : list) {
    keys(item, {"name", "endpoint", "defaults", "cases", "concurrency", "arrival_rates",
                "sessions", "warmup_sessions", "repetitions"}, "scenario");
    Scenario s;
    s.name = item.at("name").get<std::string>();
    require(!s.name.empty() && names.insert(s.name).second, "scenario names must be nonempty and unique");
    const auto endpoint = item.value("endpoint", std::string("chat"));
    require(endpoint == "chat" || endpoint == "text", "endpoint must be chat or text");
    s.chat = endpoint == "chat";
    s.sessions = integer(item.value("sessions", Json(16)), 1, 1000000, "sessions");
    s.warmups = integer(item.value("warmup_sessions", Json(2)), 0, 1000000, "warmup_sessions");
    s.repetitions = integer(item.value("repetitions", Json(3)), 1, 10000, "repetitions");
    const auto concurrency = item.value("concurrency", Json::array({1, 4, 8}));
    require(concurrency.is_array() && !concurrency.empty(), "concurrency must be a nonempty array");
    for (const auto& value : concurrency) {
      const auto n = integer(value, 1, 4096, "concurrency");
      require(std::find(s.concurrency.begin(), s.concurrency.end(), n) == s.concurrency.end(), "duplicate concurrency");
      s.concurrency.push_back(n);
    }
    const auto rates = item.value("arrival_rates", Json::array({0}));
    require(rates.is_array() && !rates.empty(), "arrival_rates must be a nonempty array");
    for (const auto& value : rates) {
      const double rate = number(value, 0, 1000000, "arrival rate");
      require(rate == 0 || rate >= .001, "positive arrival rates must be at least .001 sessions/second");
      require(std::find(s.arrival_rates.begin(), s.arrival_rates.end(), rate) == s.arrival_rates.end(), "duplicate arrival rate");
      s.arrival_rates.push_back(rate);
    }
    const auto defaults = item.value("defaults", Json::object());
    require(defaults.is_object(), "defaults must be an object");
    const auto& cases = item.at("cases");
    require(cases.is_array() && !cases.empty(), "cases must be a nonempty array");
    std::set<std::string> case_names;
    for (const auto& input : cases) {
      keys(input, {"name", "body", "followups", "cancel_after_ms", "receive_bytes_per_second"}, "case");
      Case c;
      c.name = input.at("name").get<std::string>();
      require(!c.name.empty() && case_names.insert(c.name).second, "case names must be nonempty and unique");
      c.body = defaults;
      require(input.at("body").is_object(), "case body must be an object");
      c.body.update(input.at("body"));
      require(!c.body.contains("model"), "set model with --model, not in request bodies");
      require(!c.body.contains("n") || integer(c.body.at("n"), 1, 1, "n") == 1, "only n=1 is supported");
      if (!c.body.contains("stream")) c.body["stream"] = true;
      require(c.body["stream"].is_boolean(), "stream must be boolean");
      if (c.body["stream"].get<bool>()) {
        if (!c.body.contains("stream_options")) c.body["stream_options"] = Json::object();
        require(c.body["stream_options"].is_object(), "stream_options must be an object");
        require(!c.body["stream_options"].contains("include_usage") ||
                c.body["stream_options"]["include_usage"] == true, "streaming benchmarks require include_usage=true");
        c.body["stream_options"]["include_usage"] = true;
      } else require(!c.body.contains("stream_options"), "stream_options requires streaming");
      if (s.chat) require(c.body.contains("messages") && c.body["messages"].is_array() && !c.body["messages"].empty() &&
                          !c.body.contains("prompt"), "chat cases require messages and no prompt");
      else require(c.body.contains("prompt") && c.body["prompt"].is_string() && !c.body.contains("messages"),
                   "text cases require a string prompt and no messages");
      c.followups = input.value("followups", Json::array());
      require(c.followups.is_array() && c.followups.size() <= 1024 && (s.chat || c.followups.empty()),
              "followups must be a chat-only array of at most 1024 messages");
      for (const auto& followup : c.followups)
        require(followup.is_object() && followup.value("role", std::string()) == "user" && followup.contains("content"),
                "each followup must be a user message");
      c.cancel_after = number(input.value("cancel_after_ms", Json(0)), 0, 86400000, "cancel_after_ms") / 1000;
      c.receive_rate = integer(input.value("receive_bytes_per_second", Json(0)), 0, 1000000000, "receive_bytes_per_second");
      s.cases.push_back(std::move(c));
    }
    for (const auto count : {s.sessions, s.warmups}) {
      std::uint64_t requests = 0;
      for (std::size_t i = 0; i < count; ++i) requests += 1 + s.cases[i % s.cases.size()].followups.size();
      require(requests <= 1000000, "each phase is limited to one million planned requests");
    }
    if (options.scenario.empty() || options.scenario == s.name) result.push_back(std::move(s));
  }
  require(!result.empty(), "no matching scenario");
  return result;
}

// Decodes the public response only. Token counts never come from text or SSE frames.
struct Response {
  bool chat, streaming, done = false, first_line = true, previous_cr = false;
  std::string line, event, buffered, finish_reason, text, refusal;
  bool has_data = false;
  Json usage = nullptr, message = Json::object();
  std::map<std::size_t, Json> calls;
  std::optional<Time> first_content, first_event, last_content;
  std::size_t content_events = 0;
  double max_content_gap = 0;

  Response(bool is_chat, bool is_streaming) : chat(is_chat), streaming(is_streaming) {}

  void content(const Json& value, Time now, std::string& destination) {
    if (value.is_null()) return;
    require(value.is_string(), "response content must be a string or null");
    const auto& fragment = value.get_ref<const std::string&>();
    require(fragment.empty() || finish_reason.empty(), "content arrived after finish_reason");
    destination += fragment;
    if (streaming && !fragment.empty()) {
      if (!first_content) first_content = now;
      if (last_content) max_content_gap = std::max(max_content_gap, seconds(now, *last_content));
      last_content = now;
      ++content_events;
    }
  }
  void consume(const Json& value, Time now) {
    require(value.is_object(), "response event must be an object");
    if (value.contains("error") && !value["error"].is_null())
      throw std::runtime_error("server error: " + value["error"].dump());
    if (!first_event) first_event = now;
    if (value.contains("usage") && !value["usage"].is_null()) {
      require(usage.is_null() || usage == value["usage"], "conflicting usage records");
      usage = value["usage"];
    }
    const auto choices = value.value("choices", Json::array());
    require(choices.is_array() && choices.size() <= 1, "expected zero or one response choice");
    if (choices.empty()) return;
    const auto& choice = choices[0];
    require(choice.is_object(), "choice must be an object");
    integer(choice.value("index", Json(0)), 0, 0, "choice index");
    if (chat) {
      const auto delta = choice.value(streaming ? "delta" : "message", Json::object());
      require(delta.is_object(), "chat delta/message must be an object");
      if (delta.contains("content")) content(delta["content"], now, text);
      if (delta.contains("refusal")) content(delta["refusal"], now, refusal);
      if (delta.contains("tool_calls") && !delta["tool_calls"].is_null()) {
        require(delta["tool_calls"].is_array(), "tool_calls must be an array");
        require(delta["tool_calls"].empty() || finish_reason.empty(), "tool call arrived after finish_reason");
        for (const auto& call : delta["tool_calls"]) {
          require(call.is_object(), "tool call must be an object");
          const auto index = streaming ? integer(call.at("index"), 0, 1023, "tool index") : calls.size();
          auto& target = calls[index];
          if (target.is_null()) target = {{"type", "function"}, {"function", {{"name", ""}, {"arguments", ""}}}};
          if (call.contains("id")) target["id"] = call.at("id").get<std::string>();
          if (call.contains("type")) require(call["type"] == "function", "unsupported response tool type");
          if (call.contains("function")) {
            require(call["function"].is_object(), "tool function must be an object");
            for (const auto* field : {"name", "arguments"})
              if (call["function"].contains(field))
                target["function"][field] = target["function"][field].get<std::string>() +
                                             call["function"][field].get<std::string>();
          }
        }
      }
    } else if (choice.contains("text")) content(choice["text"], now, text);
    if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
      const auto reason = choice["finish_reason"].get<std::string>();
      require(!reason.empty() && (finish_reason.empty() || finish_reason == reason), "invalid/conflicting finish_reason");
      finish_reason = reason;
    }
  }
  void end_line(Time now) {
    if (first_line) {
      if (line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
      first_line = false;
    }
    if (line.empty()) {
      if (has_data) {
        require(!done, "response event after [DONE]");
        if (!event.empty()) event.pop_back();
        if (event == "[DONE]") done = true;
        else consume(Json::parse(event), now);
        event.clear();
        has_data = false;
      }
    } else if (line[0] != ':') {
      const auto colon = line.find(':');
      if (line.substr(0, colon) == "data") {
        auto value = colon == std::string::npos ? std::string() : line.substr(colon + 1);
        if (!value.empty() && value[0] == ' ') value.erase(0, 1);
        event += value + '\n';
        has_data = true;
      }
    }
    line.clear();
  }
  void feed(const char* data, std::size_t size, Time now) {
    if (!streaming) { buffered.append(data, size); return; }
    for (std::size_t i = 0; i < size; ++i) {
      const char c = data[i];
      if (c == '\n' && previous_cr) { previous_cr = false; continue; }
      previous_cr = c == '\r';
      if (c == '\n' || c == '\r') end_line(now);
      else line.push_back(c);
    }
  }
  void complete(Time now) {
    if (streaming) {
      require(done, "truncated SSE: missing terminated [DONE] event");
      require(line.empty() && !has_data, "unterminated trailing SSE event");
    } else consume(Json::parse(buffered), now);
    require(!finish_reason.empty(), "missing finish_reason");
    if (!usage.is_null()) {
      require(usage.is_object(), "usage must be an object");
      for (const auto* key : {"prompt_tokens", "completion_tokens"})
        integer(usage.at(key), 0, std::numeric_limits<std::int64_t>::max(), std::string("usage.") + key);
      if (usage.contains("prompt_tokens_details") && !usage["prompt_tokens_details"].is_null()) {
        const auto& details = usage["prompt_tokens_details"];
        require(details.is_object(), "prompt_tokens_details must be an object");
        if (details.contains("cached_tokens") && !details["cached_tokens"].is_null())
          integer(details["cached_tokens"], 0, usage["prompt_tokens"].get<std::uint64_t>(), "cached_tokens");
      }
    }
    message = {{"role", "assistant"}, {"content", text.empty() && !calls.empty() ? Json(nullptr) : Json(text)}};
    if (!refusal.empty()) message["refusal"] = refusal;
    if (!calls.empty()) {
      message["tool_calls"] = Json::array();
      for (const auto& [index, call] : calls) {
        require(index == message["tool_calls"].size() && call.contains("id") &&
                !call["function"]["name"].get<std::string>().empty(), "incomplete tool call");
        message["tool_calls"].push_back(call);
      }
    }
  }
};

template<class T> void option(CURL* curl, CURLoption key, T value) {
  const auto code = curl_easy_setopt(curl, key, value);
  require(code == CURLE_OK, std::string("curl option: ") + curl_easy_strerror(code));
}
void multi_check(CURLMcode code) {
  require(code == CURLM_OK, std::string("curl multi: ") + curl_multi_strerror(code));
}

struct Session {
  std::size_t id, turn = 0;
  const Case* input;
  Json body;
  Time scheduled, started;
};
struct Transfer {
  CURL* easy = curl_easy_init();
  curl_slist* headers = nullptr;
  Session session;
  Response response;
  std::string payload, request_hash, error, error_body;
  char curl_error[CURL_ERROR_SIZE]{};
  std::size_t response_bytes = 0, limit;
  long http_status = 0;

  Transfer(Session next, bool chat, const Options& options)
      : session(std::move(next)), response(chat, session.body.at("stream").get<bool>()),
        payload(session.body.dump()), request_hash(sha256(payload)), limit(options.max_response_bytes) {
    require(easy != nullptr, "cannot allocate curl handle");
    try {
      const std::string url = options.base_url + (chat ? "/chat/completions" : "/completions");
      option(easy, CURLOPT_URL, url.c_str());
      option(easy, CURLOPT_PROTOCOLS_STR, "http,https");
      option(easy, CURLOPT_NOPROXY, "*");
      option(easy, CURLOPT_NOSIGNAL, 1L);
      option(easy, CURLOPT_HTTP_VERSION, long(CURL_HTTP_VERSION_1_1));
      option(easy, CURLOPT_TIMEOUT_MS, long(std::ceil(options.timeout * 1000)));
      option(easy, CURLOPT_CONNECTTIMEOUT_MS, long(std::ceil(options.connect_timeout * 1000)));
      option(easy, CURLOPT_POSTFIELDS, payload.c_str());
      option(easy, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(payload.size()));
      option(easy, CURLOPT_ERRORBUFFER, curl_error);
      option(easy, CURLOPT_WRITEFUNCTION, &Transfer::write);
      option(easy, CURLOPT_WRITEDATA, this);
      option(easy, CURLOPT_HEADERFUNCTION, &Transfer::header);
      option(easy, CURLOPT_HEADERDATA, this);
      if (session.input->receive_rate)
        option(easy, CURLOPT_MAX_RECV_SPEED_LARGE, session.input->receive_rate);
      add_header("Content-Type: application/json");
      add_header(response.streaming ? "Accept: text/event-stream" : "Accept: application/json");
      add_header("Expect:");
      if (!options.api_key.empty()) add_header("Authorization: Bearer " + options.api_key);
      option(easy, CURLOPT_HTTPHEADER, headers);
    } catch (...) { cleanup(); throw; }
  }
  ~Transfer() { cleanup(); }
  void cleanup() {
    if (easy) curl_easy_cleanup(easy);
    curl_slist_free_all(headers);
    easy = nullptr; headers = nullptr;
  }
  void add_header(const std::string& value) {
    auto* next = curl_slist_append(headers, value.c_str());
    require(next != nullptr, "cannot allocate HTTP header");
    headers = next;
  }
  static std::size_t header(char* bytes, std::size_t size, std::size_t count, void* context) noexcept {
    auto& self = *static_cast<Transfer*>(context);
    try {
      const std::string line(bytes, size * count);
      if (line.rfind("HTTP/", 0) == 0) {
        const auto space = line.find(' ');
        require(space != std::string::npos, "invalid HTTP status line");
        self.http_status = std::stol(line.substr(space + 1));
      }
      return size * count;
    } catch (const std::exception& e) { self.error = e.what(); return 0; }
  }
  static std::size_t write(char* bytes, std::size_t size, std::size_t count, void* context) noexcept {
    auto& self = *static_cast<Transfer*>(context);
    try {
      const auto n = size * count;
      require(n <= self.limit - self.response_bytes, "response exceeds --max-response-bytes");
      self.response_bytes += n;
      if (self.http_status >= 200 && self.http_status < 300) self.response.feed(bytes, n, Clock::now());
      else if (self.error_body.size() < 4096) self.error_body.append(bytes, std::min(n, 4096 - self.error_body.size()));
      return n;
    } catch (const std::exception& e) { self.error = e.what(); return 0; }
  }
};

Json distribution(std::vector<double> values) {
  if (values.empty()) return nullptr;
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    const double at = p * (values.size() - 1);
    const auto lo = static_cast<std::size_t>(at);
    return values[lo] + (values[std::min(lo + 1, values.size() - 1)] - values[lo]) * (at - lo);
  };
  double total = 0;
  for (double value : values) total += value;
  return {{"count", values.size()}, {"min", values.front()}, {"p50", percentile(.5)},
          {"p90", percentile(.9)}, {"p95", percentile(.95)}, {"p99", percentile(.99)},
          {"max", values.back()}, {"mean", total / values.size()}};
}
struct Statistics {
  std::uint64_t completed = 0, failed = 0, cancelled = 0, with_usage = 0, prompt_tokens = 0, output_tokens = 0;
  std::size_t peak_active = 0;
  Json finishes = Json::object(), statuses = Json::object();
  std::vector<double> latency, http, first_content, queue, max_content_gap;
  void add(const Json& record) {
    const std::string status = record.at("status");
    const auto code = std::to_string(record.at("http_status").get<long>());
    statuses[code] = statuses.value(code, 0ULL) + 1;
    queue.push_back(record.at("queue_seconds"));
    if (status == "cancelled") { ++cancelled; return; }
    if (status == "failed") { ++failed; return; }
    ++completed;
    latency.push_back(record.at("latency_seconds"));
    http.push_back(record.at("http_seconds"));
    if (!record.at("first_content_seconds").is_null()) {
      first_content.push_back(record.at("first_content_seconds"));
      max_content_gap.push_back(record.at("max_content_gap_seconds"));
    }
    const std::string reason = record.at("finish_reason");
    finishes[reason] = finishes.value(reason, 0ULL) + 1;
    if (!record.at("usage").is_null()) {
      ++with_usage;
      prompt_tokens += record["usage"]["prompt_tokens"].get<std::uint64_t>();
      output_tokens += record["usage"]["completion_tokens"].get<std::uint64_t>();
    }
  }
  Json summary(double wall, double cpu) const {
    return {{"wall_seconds", wall}, {"client_cpu_seconds", cpu}, {"completed", completed}, {"failed", failed},
            {"cancelled", cancelled}, {"peak_active_requests", peak_active}, {"requests_with_usage", with_usage},
            {"known_prompt_tokens", prompt_tokens}, {"known_completion_tokens", output_tokens},
            {"completed_requests_per_second", wall > 0 ? Json(completed / wall) : Json(nullptr)},
            {"completion_tokens_per_second", completed > 0 && completed == with_usage && wall > 0 ? Json(output_tokens / wall) : Json(nullptr)},
            {"latency_seconds", distribution(latency)}, {"http_seconds", distribution(http)},
            {"first_content_seconds", distribution(first_content)}, {"queue_seconds", distribution(queue)},
            {"max_content_gap_seconds", distribution(max_content_gap)},
            {"finish_reasons", finishes}, {"http_statuses", statuses}};
  }
};

Json execute_phase(CURLM* multi, const Scenario& scenario, std::size_t concurrency, double rate,
                   std::size_t sessions, const Options& options, const Json& identity, std::ofstream& records) {
  std::map<CURL*, std::unique_ptr<Transfer>> active;
  std::size_t next_session = 0;
  Statistics stats;
  const auto cpu_start = std::clock();
  const Time start = Clock::now();
  auto launch = [&](Session session) {
    auto transfer = std::make_unique<Transfer>(std::move(session), scenario.chat, options);
    // Include request serialization/handle preparation in queue delay, not HTTP time.
    transfer->session.started = Clock::now();
    auto* handle = transfer->easy;
    multi_check(curl_multi_add_handle(multi, handle));
    active.emplace(handle, std::move(transfer));
    stats.peak_active = std::max(stats.peak_active, active.size());
  };
  auto finish = [&](CURL* handle, CURLcode code, bool cancelled) {
    auto found = active.find(handle);
    require(found != active.end(), "unknown completed HTTP transfer");
    auto transfer = std::move(found->second);
    active.erase(found);
    auto& t = *transfer;
    const auto end = Clock::now();
    multi_check(curl_multi_remove_handle(multi, handle));
    std::string status = cancelled ? "cancelled" : "completed";
    if (!cancelled) {
      try {
        require(t.error.empty(), t.error);
        require(code == CURLE_OK, t.curl_error[0] ? t.curl_error : curl_easy_strerror(code));
        require(t.http_status >= 200 && t.http_status < 300,
                "HTTP " + std::to_string(t.http_status) + ": " + t.error_body);
        char* content_type = nullptr;
        curl_easy_getinfo(handle, CURLINFO_CONTENT_TYPE, &content_type);
        std::string type = content_type ? content_type : "";
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        require(type.rfind(t.response.streaming ? "text/event-stream" : "application/json", 0) == 0,
                "unexpected response Content-Type");
        t.response.complete(end);
        if (t.session.turn < t.session.input->followups.size())
          require(t.response.calls.empty() && !t.response.text.empty(),
                  "growing conversations require an assistant text answer; tool execution is not implemented");
      } catch (const std::exception& e) { status = "failed"; t.error = e.what(); }
    }
    const auto from_scheduled = [&](const std::optional<Time>& time) -> Json {
      return time ? Json(seconds(*time, t.session.scheduled)) : Json(nullptr);
    };
    Json record = identity;
    record.update({{"session", t.session.id}, {"turn", t.session.turn}, {"case", t.session.input->name},
                   {"status", status}, {"http_status", t.http_status}, {"error", t.error.empty() ? Json(nullptr) : Json(t.error)},
                   {"scheduled_seconds", seconds(t.session.scheduled, start)},
                   {"started_seconds", seconds(t.session.started, start)}, {"finished_seconds", seconds(end, start)},
                   {"queue_seconds", seconds(t.session.started, t.session.scheduled)},
                   {"latency_seconds", seconds(end, t.session.scheduled)}, {"http_seconds", seconds(end, t.session.started)},
                   {"first_content_seconds", from_scheduled(t.response.first_content)},
                   {"first_event_seconds", from_scheduled(t.response.first_event)},
                   {"content_events", t.response.content_events}, {"max_content_gap_seconds", t.response.max_content_gap},
                   {"request_sha256", t.request_hash}, {"request_bytes", t.payload.size()}, {"response_bytes", t.response_bytes},
                   {"finish_reason", t.response.finish_reason}, {"usage", status == "completed" ? t.response.usage : Json(nullptr)},
                   {"text", t.response.text}, {"message", t.response.message}});
    // HTTP error bodies may be arbitrary bytes. Preserve their readable portion
    // without letting invalid UTF-8 prevent the remaining requests from running.
    records << record.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    require(bool(records), "cannot write requests.jsonl");
    stats.add(record);
    if (status == "completed" && t.session.turn < t.session.input->followups.size()) {
      t.session.body["messages"].push_back(t.response.message);
      t.session.body["messages"].push_back(t.session.input->followups[t.session.turn]);
      ++t.session.turn;
      t.session.scheduled = end;
      launch(std::move(t.session));
    }
  };
  try {
    while (next_session < sessions || !active.empty()) {
      while (next_session < sessions && active.size() < concurrency) {
        const auto now = Clock::now();
        const auto scheduled = rate > 0 ? after(start, next_session / rate) : now;
        if (scheduled > now) break;
        const auto& input = scenario.cases[next_session % scenario.cases.size()];
        Json body = input.body;
        body["model"] = options.model;
        launch({next_session++, 0, &input, std::move(body), scheduled, now});
      }
      int running = 0;
      multi_check(curl_multi_perform(multi, &running));
      int remaining = 0;
      while (auto* message = curl_multi_info_read(multi, &remaining)) {
        if (message->msg == CURLMSG_DONE) finish(message->easy_handle, message->data.result, false);
      }
      std::vector<CURL*> cancel;
      const auto now = Clock::now();
      for (const auto& [handle, t] : active)
        if (t->session.input->cancel_after > 0 && seconds(now, t->session.started) >= t->session.input->cancel_after)
          cancel.push_back(handle);
      for (auto* handle : cancel) finish(handle, CURLE_ABORTED_BY_CALLBACK, true);
      if (next_session == sessions && active.empty()) break;
      if (next_session < sessions && active.size() < concurrency &&
          (rate == 0 || after(start, next_session / rate) <= Clock::now())) continue;
      double wait = .05;
      if (next_session < sessions && active.size() < concurrency && rate > 0)
        wait = std::min(wait, std::max(0.0, seconds(after(start, next_session / rate), Clock::now())));
      for (const auto& [handle, t] : active) {
        (void)handle;
        if (t->session.input->cancel_after > 0)
          wait = std::min(wait, std::max(0.0, t->session.input->cancel_after - seconds(Clock::now(), t->session.started)));
      }
      int ready = 0;
      multi_check(curl_multi_poll(multi, nullptr, 0, static_cast<int>(std::ceil(wait * 1000)), &ready));
    }
  } catch (...) {
    for (const auto& [handle, t] : active) { (void)t; curl_multi_remove_handle(multi, handle); }
    throw;
  }
  records.flush();
  require(bool(records), "cannot flush requests.jsonl");
  return stats.summary(seconds(Clock::now(), start), double(std::clock() - cpu_start) / CLOCKS_PER_SEC);
}

int run(const Options& options) {
  const auto source = read_file(options.scenario_file);
  const auto document = Json::parse(source);
  const auto selected = scenarios(document, options);
  Json matrix = Json::array();
  for (const auto& scenario : selected)
    for (auto concurrency : scenario.concurrency)
      for (double rate : scenario.arrival_rates)
        for (std::size_t repetition = 0; repetition < scenario.repetitions; ++repetition)
          matrix.push_back({{"scenario", scenario.name}, {"concurrency", concurrency},
                            {"arrival_rate", rate}, {"repetition", repetition},
                            {"sessions", scenario.sessions}, {"warmup_sessions", scenario.warmups}});
  if (options.plan) { std::cout << matrix.dump(2) << '\n'; return 0; }
  require(fs::create_directory(options.output), "output directory already exists; use a new directory");
  const fs::path output = options.output;
  write_json(output / "manifest.json", {{"format_version", 1}, {"client_version", GEWELL_BENCH_VERSION},
      {"client_source_sha256", GEWELL_BENCH_SOURCE_SHA256}, {"client_build", GEWELL_BENCH_BUILD},
      {"curl_version", curl_version()}, {"base_url", options.base_url}, {"model", options.model},
      {"label", options.label}, {"authenticated", !options.api_key.empty()},
      {"scenario_sha256", sha256(source)}, {"scenario_document", document}, {"matrix", matrix},
      {"timeout_seconds", options.timeout}, {"connect_timeout_seconds", options.connect_timeout},
      {"max_response_bytes", options.max_response_bytes}, {"server_cache_reset", false},
      {"clock", "steady_clock"}, {"http_version", "1.1"}});
  std::ofstream records(output / "requests.jsonl");
  require(bool(records), "cannot create requests.jsonl");
  Json report = {{"format_version", 1}, {"complete", false}, {"points", Json::array()}};
  std::uint64_t failures = 0;
  write_json(output / "summary.json", report);
  for (const auto& point : matrix) {
    const auto found = std::find_if(selected.begin(), selected.end(), [&](const Scenario& value) { return value.name == point["scenario"]; });
    const auto& scenario = *found;
    std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)> multi(curl_multi_init(), curl_multi_cleanup);
    require(bool(multi), "cannot create curl multi handle");
    Json result = point;
    Json identity = point;
    identity["phase"] = "warmup";
    result["warmup"] = execute_phase(multi.get(), scenario, point["concurrency"], point["arrival_rate"],
                                      scenario.warmups, options, identity, records);
    identity["phase"] = "measured";
    result["measured"] = execute_phase(multi.get(), scenario, point["concurrency"], point["arrival_rate"],
                                        scenario.sessions, options, identity, records);
    failures += result["warmup"]["failed"].get<std::uint64_t>() + result["measured"]["failed"].get<std::uint64_t>();
    report["points"].push_back(result);
    report["unexpected_failures"] = failures;
    write_json(output / "summary.json", report);
    const auto& stats = result["measured"];
    std::cout << scenario.name << " concurrency=" << point["concurrency"] << " rate=" << point["arrival_rate"]
              << " repeat=" << point["repetition"] << " completed=" << stats["completed"]
              << " failed=" << stats["failed"] << " cancelled=" << stats["cancelled"]
              << " output_tok/s=" << stats["completion_tokens_per_second"] << '\n';
  }
  report["complete"] = true;
  write_json(output / "summary.json", report);
  return failures ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage; return 0; }
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "gewell-http-bench " << GEWELL_BENCH_VERSION << '\n'; return 0;
  }
  if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) { std::cerr << "curl initialization failed\n"; return 2; }
  int status;
  try { status = run(arguments(argc, argv)); }
  catch (const std::exception& error) { std::cerr << "gewell-http-bench: " << error.what() << '\n'; status = 2; }
  curl_global_cleanup();
  return status;
}
