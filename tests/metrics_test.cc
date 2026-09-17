#include "gewell/metrics.h"

#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct CommaDecimal : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
};

}  // namespace

int main() {
  try {
    gewell::metrics::Histogram histogram({0.1, 0.5, 1});
    for (double value : {0., 0.1, 0.2, 2.}) histogram.observe(value);
    require(histogram.buckets == std::vector<std::uint64_t>{2, 3, 3}, "histogram boundary/cumulative count mismatch");
    require(histogram.count == 4 && histogram.sum == 2.3, "histogram sum/count mismatch");
    bool rejected = false;
    try { histogram.observe(std::numeric_limits<double>::quiet_NaN()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && histogram.count == 4, "invalid observation mutated histogram");

    const auto locale = std::locale();
    std::locale::global(std::locale(locale, new CommaDecimal));
    gewell::metrics::Writer out("a\"b\\c\nd");
    out.histogram("test:latency", "Help\\line\nnext", histogram);
    out.counter("test:tokens_total", "Tokens.", std::uint64_t{18446744073709551615ULL}, {{"source", "one\"two"}});
    out.counter("test:tokens_total", "Tokens.", std::uint64_t{2}, {{"source", "other"}});
    std::locale::global(locale);
    const auto& text = out.str();
    require(text.find("# HELP test:latency Help\\\\line\\nnext\n") != std::string::npos, "HELP escaping failed");
    require(text.find("test:latency_bucket{model_name=\"a\\\"b\\\\c\\nd\",le=\"0.1\"} 2\n") != std::string::npos,
            "label escaping, decimal formatting or inclusive bucket failed");
    require(text.find("le=\"+Inf\"} 4\n") != std::string::npos, "+Inf bucket must equal count");
    require(text.find("} 18446744073709551615\n") != std::string::npos, "counter precision lost");
    const auto type = text.find("# TYPE test:tokens_total counter\n");
    require(type != std::string::npos && text.find("# TYPE test:tokens_total", type + 1) == std::string::npos,
            "repeated labels duplicated family metadata");

    gewell::metrics::Snapshot snapshot;
    snapshot.mtp_depth = 3;
    snapshot.mtp_accepted_per_position = {4, 2};
    snapshot.prompt_tokens = 12;
    snapshot.computed_prompt_tokens = 7;
    snapshot.cached_prompt_tokens = 5;
    snapshot.shared_prompt_tokens = 2;
    snapshot.success_length = 1;
    snapshot.ttft.observe(0.04);
    const auto rendered = snapshot.render("fixture");
    require(rendered.find("vllm:prompt_tokens_total{model_name=\"fixture\"} 12\n") != std::string::npos, "prompt total missing");
    require(rendered.find("source=\"local_compute\"} 7\n") != std::string::npos, "computed source missing");
    require(rendered.find("source=\"local_cache_hit\"} 5\n") != std::string::npos, "cached source missing");
    require(rendered.find("vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"fixture\",position=\"2\"} 0\n") != std::string::npos,
            "unobserved configured draft positions must start at zero");
    require(rendered.find("vllm:time_to_first_token_seconds_count{model_name=\"fixture\"} 1\n") != std::string::npos,
            "snapshot histogram count missing");
    require(rendered.find("vllm:request_queue_time_seconds_count{model_name=\"fixture\"} 0\n") != std::string::npos,
            "zero histogram series missing");
    require(!rendered.empty() && rendered.back() == '\n', "metrics require trailing newline");
    std::cout << "metrics tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
