#include "gewell/metrics.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>

namespace gewell::metrics {
namespace {

std::string escape(std::string_view value, bool label) {
  std::string result;
  for (const char ch : value) {
    if (ch == '\\') result += "\\\\";
    else if (ch == '\n') result += "\\n";
    else if (ch == '"' && label) result += "\\\"";
    else result += ch;
  }
  return result;
}

std::string number(double value) {
  if (std::isnan(value)) return "NaN";
  if (std::isinf(value)) return value < 0 ? "-Inf" : "+Inf";
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{}) throw std::runtime_error("cannot format metric value");
  return {buffer, result.ptr};
}

std::vector<double> latency_bounds() {
  return {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 1.5,
          2, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 120, 240, 480, 960, 1920, 7680};
}

std::vector<double> token_latency_bounds() {
  return {0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
          0.4, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 80};
}

std::vector<double> length_bounds() {
  return {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
          20000, 50000, 100000, 200000, 262144};
}

}  // namespace

Histogram::Histogram(std::vector<double> finite_bounds)
    : bounds(std::move(finite_bounds)), buckets(bounds.size(), 0) {
  if (bounds.empty() || !std::is_sorted(bounds.begin(), bounds.end()) ||
      std::adjacent_find(bounds.begin(), bounds.end()) != bounds.end() ||
      std::any_of(bounds.begin(), bounds.end(), [](double v) { return !std::isfinite(v) || v < 0; }))
    throw std::invalid_argument("histogram bounds must be finite, nonnegative and strictly increasing");
}

void Histogram::observe(double value) {
  if (!std::isfinite(value) || value < 0)
    throw std::invalid_argument("histogram observations must be finite and nonnegative");
  ++count;
  sum += value;
  const auto begin = std::lower_bound(bounds.begin(), bounds.end(), value) - bounds.begin();
  for (std::size_t i = begin; i < buckets.size(); ++i) ++buckets[i];
}

Writer::Writer(std::string model_name) : model_name_(escape(model_name, true)) {}

void Writer::family(std::string_view name, std::string_view help, std::string_view type) {
  const auto inserted = families_.emplace(std::string(name), std::string(type));
  if (!inserted.second) {
    if (inserted.first->second != type) throw std::logic_error("conflicting metric family types");
    return;
  }
  output_ += "# HELP " + std::string(name) + " " + escape(help, false) + "\n# TYPE " +
      std::string(name) + " " + std::string(type) + "\n";
}

void Writer::sample(std::string_view name, std::string_view value, const Labels& labels) {
  output_ += std::string(name) + "{model_name=\"" + model_name_ + "\"";
  for (const auto& [key, text] : labels) output_ += "," + key + "=\"" + escape(text, true) + "\"";
  output_ += "} " + std::string(value) + "\n";
}

void Writer::gauge(std::string_view name, std::string_view help, double value, const Labels& labels) {
  family(name, help, "gauge");
  sample(name, number(value), labels);
}

void Writer::counter(std::string_view name, std::string_view help, std::uint64_t value, const Labels& labels) {
  family(name, help, "counter");
  sample(name, std::to_string(value), labels);
}

void Writer::counter(std::string_view name, std::string_view help, double value, const Labels& labels) {
  family(name, help, "counter");
  sample(name, number(value), labels);
}

void Writer::histogram(std::string_view name, std::string_view help,
                       const Histogram& value, const Labels& labels) {
  family(name, help, "histogram");
  auto bucket_labels = labels;
  bucket_labels.emplace_back("le", "");
  const auto bucket_name = std::string(name) + "_bucket";
  for (std::size_t i = 0; i < value.bounds.size(); ++i) {
    bucket_labels.back().second = number(value.bounds[i]);
    sample(bucket_name, std::to_string(value.buckets[i]), bucket_labels);
  }
  bucket_labels.back().second = "+Inf";
  sample(bucket_name, std::to_string(value.count), bucket_labels);
  sample(std::string(name) + "_sum", number(value.sum), labels);
  sample(std::string(name) + "_count", std::to_string(value.count), labels);
}

Snapshot::Snapshot()
    : ttft({0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
            0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 80, 160, 640, 2560}),
      e2e(latency_bounds()), queue(latency_bounds()), prefill(latency_bounds()),
      decode(latency_bounds()), inference(latency_bounds()), tpot(token_latency_bounds()),
      itl(token_latency_bounds()), prompt_length(length_bounds()), generation_length(length_bounds()) {}

std::string Snapshot::render(const std::string& model_name) const {
  Writer out(model_name);
  out.gauge("vllm:num_requests_running", "Generation requests admitted to execution, including prefill.", running);
  out.gauge("vllm:num_requests_waiting", "Generation requests waiting for first scheduling.", waiting);
  out.counter("vllm:prompt_tokens_total", "Prompt tokens processed, including cache reuse and shared work.", prompt_tokens);
  out.counter("vllm:prompt_tokens_by_source_total", "Prompt tokens by source.", computed_prompt_tokens,
              {{"source", "local_compute"}});
  out.counter("vllm:prompt_tokens_by_source_total", "Prompt tokens by source.", cached_prompt_tokens,
              {{"source", "local_cache_hit"}});
  out.counter("gewell:shared_prompt_tokens_total", "Prompt tokens supplied by shared in-flight execution, included in local cache hits.", shared_prompt_tokens);
  out.counter("vllm:generation_tokens_total", "Generated output tokens, including EOS and excluding rejected drafts.", generation_tokens);
  out.counter("vllm:request_success_total", "Successfully completed generation requests.", success_stop,
              {{"finished_reason", "stop"}});
  out.counter("vllm:request_success_total", "Successfully completed generation requests.", success_length,
              {{"finished_reason", "length"}});
  out.counter("gewell:request_aborted_total", "Generation requests cancelled after scheduler admission.", aborted);
  out.counter("gewell:request_failed_total", "Prepared generation requests rejected by admission or failed during execution.", failed);
  out.counter("vllm:prefix_cache_queries_total", "Prompt tokens queried for prefix reuse.", prefix_queries);
  out.counter("vllm:prefix_cache_hits_total", "Prompt tokens reused from retained checkpoints or shared in-flight execution.", prefix_hits);
  out.histogram("vllm:time_to_first_token_seconds", "Time from request arrival to first output token.", ttft);
  out.histogram("vllm:e2e_request_latency_seconds", "Time from complete HTTP request body to last output token, excluding final response processing.", e2e);
  out.histogram("vllm:request_queue_time_seconds", "Time from scheduler submission to first scheduling.", queue);
  out.histogram("vllm:request_prefill_time_seconds", "Time from first scheduling to first output token.", prefill);
  out.histogram("vllm:request_decode_time_seconds", "Time from first output token to last output token.", decode);
  out.histogram("vllm:request_inference_time_seconds", "Time from first scheduling to last output token.", inference);
  out.histogram("vllm:request_time_per_output_token_seconds", "Per-request decode duration divided by output tokens after the first.", tpot);
  out.histogram("vllm:inter_token_latency_seconds", "Time between output bursts; speculative bursts count as one observation.", itl);
  out.histogram("vllm:request_prompt_tokens", "Prompt token count per completed generation request.", prompt_length);
  out.histogram("vllm:request_generation_tokens", "Output token count per completed generation request.", generation_length);
  out.counter("vllm:spec_decode_num_drafts_total", "Speculative decoding rounds.", mtp_rounds);
  out.counter("vllm:spec_decode_num_draft_tokens_total", "Tokens proposed for speculative verification.", mtp_draft_tokens);
  out.counter("vllm:spec_decode_num_accepted_tokens_total", "Draft tokens accepted by the verifier before output termination trimming.", mtp_accepted_tokens);
  for (std::size_t i = 0; i < std::max<std::size_t>(mtp_depth, mtp_accepted_per_position.size()); ++i)
    out.counter("vllm:spec_decode_num_accepted_tokens_per_pos_total", "Verifier-accepted tokens by zero-based draft position.",
                i < mtp_accepted_per_position.size() ? mtp_accepted_per_position[i] : std::uint64_t{0},
                {{"position", std::to_string(i)}});
  out.gauge("gewell:mtp_depth", "Configured maximum number of proposed MTP tokens per round.", mtp_depth);
  out.counter("gewell:mtp_emitted_tokens_total", "Actual output tokens from speculative rounds after termination trimming, including bonus tokens.", mtp_emitted_tokens);
  out.counter("gewell:mtp_rejected_rounds_total", "Speculative rounds with at least one rejected draft token.", mtp_rejected_rounds);
  out.counter("gewell:mtp_gpu_seconds_total", "Measured GPU time in MTP stages; excludes accepted-prefix commit and remaining decode work.", mtp_draft_seconds, {{"stage", "draft"}});
  out.counter("gewell:mtp_gpu_seconds_total", "Measured GPU time in MTP stages; excludes accepted-prefix commit and remaining decode work.", mtp_verify_seconds, {{"stage", "verify"}});
  out.counter("gewell:mtp_gpu_seconds_total", "Measured GPU time in MTP stages; excludes accepted-prefix commit and remaining decode work.", mtp_select_seconds, {{"stage", "select"}});
  out.counter("gewell:execution_gpu_seconds_total", "Measured GPU work time by phase, counted once per batch rather than per request.", prefill_gpu_seconds, {{"phase", "prefill"}});
  out.counter("gewell:execution_gpu_seconds_total", "Measured GPU work time by phase, counted once per batch rather than per request.", decode_gpu_seconds, {{"phase", "decode"}});
  return out.str();
}

}  // namespace gewell::metrics
