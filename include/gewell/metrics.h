#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gewell::metrics {

// The scheduler owns and updates observations. Rendering copies no runtime or
// GPU state and is safe on a published, immutable snapshot.
struct Histogram {
  explicit Histogram(std::vector<double> finite_bounds);
  void observe(double value);
  std::vector<double> bounds;
  std::vector<std::uint64_t> buckets;  // Cumulative counts, excluding +Inf.
  double sum = 0;
  std::uint64_t count = 0;
};

using Labels = std::vector<std::pair<std::string, std::string>>;

class Writer {
 public:
  explicit Writer(std::string model_name);
  void gauge(std::string_view name, std::string_view help, double value,
             const Labels& labels = {});
  void counter(std::string_view name, std::string_view help, std::uint64_t value,
               const Labels& labels = {});
  void counter(std::string_view name, std::string_view help, double value,
               const Labels& labels = {});
  void histogram(std::string_view name, std::string_view help,
                 const Histogram& value, const Labels& labels = {});
  const std::string& str() const { return output_; }
 private:
  void family(std::string_view name, std::string_view help, std::string_view type);
  void sample(std::string_view name, std::string_view value, const Labels& labels);
  std::string model_name_, output_;
  std::map<std::string, std::string> families_;
};

struct Snapshot {
  Snapshot();
  std::uint64_t running = 0, waiting = 0;
  std::uint64_t prompt_tokens = 0, computed_prompt_tokens = 0,
                cached_prompt_tokens = 0, shared_prompt_tokens = 0;
  std::uint64_t prefix_queries = 0, prefix_hits = 0, generation_tokens = 0;
  std::uint64_t success_stop = 0, success_length = 0, aborted = 0, failed = 0;
  Histogram ttft, e2e, queue, prefill, decode, inference, tpot, itl,
            prompt_length, generation_length;
  std::uint64_t mtp_rounds = 0, mtp_draft_tokens = 0, mtp_accepted_tokens = 0,
                mtp_emitted_tokens = 0, mtp_rejected_rounds = 0;
  std::uint32_t mtp_depth = 0;
  std::vector<std::uint64_t> mtp_accepted_per_position;
  double mtp_draft_seconds = 0, mtp_verify_seconds = 0, mtp_select_seconds = 0,
         prefill_gpu_seconds = 0, decode_gpu_seconds = 0;
  std::string render(const std::string& model_name) const;
};

}  // namespace gewell::metrics
