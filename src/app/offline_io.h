#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace gewell::app::offline {
inline constexpr std::size_t kMaxLineBytes = 1024 * 1024;
inline constexpr std::size_t kMaxEventsBytes = 16 * 1024 * 1024;
inline constexpr auto kOutputTimeout = std::chrono::seconds(30);

// One pending chunk per request. FIFO writes never block the scheduler; pump
// runs while GPU work is in flight, and cancellation discards pending bytes.
class Output {
 public:
  explicit Output(std::string path, bool allow_fifo);
  ~Output();
  Output(const Output&) = delete;
  Output& operator=(const Output&) = delete;
  void append(const void* data, std::size_t bytes);
  void pump();
  bool ready() const { return fd_ >= 0 && pending_.empty(); }
  std::size_t pending_bytes() const { return pending_.size() - offset_; }
  const std::string& error() const { return error_; }
  void complete();
  void cancel();
  void discard() { complete_ = false; cancel(); }
 private:
  std::string path_, error_;
  int fd_{-1};
  bool fifo_{}, created_{}, complete_{}, cancelled_{};
  std::vector<std::uint8_t> pending_;
  std::size_t offset_{};
  std::chrono::steady_clock::time_point progress_{std::chrono::steady_clock::now()};
};

// Nonblocking descriptors and a bounded single-line input buffer. The caller
// caps admitted requests and drains one line at a time during scheduler polls.
class Channel {
 public:
  Channel(int input, int output);
  ~Channel();
  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;
  std::optional<std::string> read_line();
  std::uint64_t event(std::string line);
  void pump();
  bool eof() const { return eof_ && input_buffer_.empty(); }
  bool empty() const { return output_.empty(); }
  bool ready() const { return output_bytes_ < kMaxEventsBytes / 2; }
  const std::string& error() const { return error_; }
  std::uint64_t written_events() const { return written_events_; }
 private:
  int input_, output_fd_, input_flags_, output_flags_;
  bool eof_{};
  std::string input_buffer_, error_;
  std::deque<std::string> output_;
  std::size_t output_bytes_{}, output_offset_{};
  std::uint64_t queued_events_{}, written_events_{};
  std::chrono::steady_clock::time_point progress_{std::chrono::steady_clock::now()};
};
}  // namespace gewell::app::offline
