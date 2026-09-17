#include "offline_io.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace gewell::app::offline {
namespace {
std::string system_error(const std::string& operation) {
  return operation + ": " + std::strerror(errno);
}
int nonblocking(int fd) {
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    throw std::runtime_error(system_error("set nonblocking descriptor"));
  return flags;
}
}  // namespace

Output::Output(std::string path, bool allow_fifo) : path_(std::move(path)) {
  if (path_.empty() || path_.find('\0') != std::string::npos)
    throw std::invalid_argument("output path must be nonempty and contain no NUL");
  struct stat info{};
  if (::lstat(path_.c_str(), &info) == 0) {
    if (!allow_fifo || !S_ISFIFO(info.st_mode))
      throw std::runtime_error("output already exists: " + path_);
    fifo_ = true;
    pump();
    return;
  }
  if (errno != ENOENT) throw std::runtime_error(system_error("inspect output " + path_));
  fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0666);
  if (fd_ < 0) throw std::runtime_error(system_error("create output " + path_));
  created_ = true;
}

Output::~Output() { cancel(); }

void Output::append(const void* data, std::size_t bytes) {
  if (complete_ || cancelled_ || !pending_.empty() || !error_.empty())
    throw std::logic_error("output has an unfinished or failed chunk");
  if (!bytes) return;
  const auto* begin = static_cast<const std::uint8_t*>(data);
  pending_.assign(begin, begin + bytes);
  offset_ = 0;
  progress_ = std::chrono::steady_clock::now();
}

void Output::pump() {
  if (complete_ || cancelled_ || !error_.empty()) return;
  if (fifo_ && fd_ < 0) {
    fd_ = ::open(path_.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
    if (fd_ < 0 && errno != ENXIO && errno != EINTR)
      error_ = system_error("open logit FIFO " + path_);
    if (fd_ >= 0) {
      struct stat info{};
      if (::fstat(fd_, &info) != 0 || !S_ISFIFO(info.st_mode))
        error_ = "logit FIFO changed before it opened: " + path_;
      progress_ = std::chrono::steady_clock::now();
    }
  }
  // A finite quantum lets other requests and cancellation keep making progress.
  if (fd_ >= 0 && !pending_.empty() && error_.empty()) {
    const auto count = ::write(fd_, pending_.data() + offset_,
        std::min<std::size_t>(pending_.size() - offset_, 1024 * 1024));
    if (count > 0) {
      offset_ += count;
      progress_ = std::chrono::steady_clock::now();
      if (offset_ == pending_.size()) { pending_.clear(); offset_ = 0; }
    } else if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      error_ = system_error("write output " + path_);
    }
  }
  if ((fd_ < 0 || !pending_.empty()) &&
      std::chrono::steady_clock::now() - progress_ > kOutputTimeout)
    error_ = "output made no progress for 30 seconds: " + path_;
}

void Output::complete() {
  if (!error_.empty()) throw std::runtime_error(error_);
  if (!ready()) throw std::logic_error("output completed before its pending bytes were written");
  const int fd = fd_;
  fd_ = -1;
  if (::close(fd) != 0) throw std::runtime_error(system_error("close output " + path_));
  complete_ = true;
}

void Output::cancel() {
  cancelled_ = true;
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  pending_.clear();
  if (created_ && !complete_) { ::unlink(path_.c_str()); created_ = false; }
}

Channel::Channel(int input, int output) : input_(input), output_fd_(output),
    input_flags_(nonblocking(input)), output_flags_(0) {
  try { output_flags_ = nonblocking(output); }
  catch (...) { ::fcntl(input_, F_SETFL, input_flags_); throw; }
}
Channel::~Channel() {
  ::fcntl(input_, F_SETFL, input_flags_);
  ::fcntl(output_fd_, F_SETFL, output_flags_);
}

std::optional<std::string> Channel::read_line() {
  for (;;) {
    const auto end = input_buffer_.find('\n');
    if (end != std::string::npos) {
      auto line = input_buffer_.substr(0, end);
      input_buffer_.erase(0, end + 1);
      return line;
    }
    if (eof_) {
      if (input_buffer_.empty()) return {};
      return std::exchange(input_buffer_, {});
    }
    if (input_buffer_.size() > kMaxLineBytes)
      throw std::runtime_error("offline job line exceeds 1 MiB");
    char buffer[4096];
    const auto count = ::read(input_, buffer,
        std::min(sizeof(buffer), kMaxLineBytes + 1 - input_buffer_.size()));
    if (count > 0) input_buffer_.append(buffer, count);
    else if (count == 0) eof_ = true;
    else if (errno == EINTR) continue;
    else if (errno == EAGAIN || errno == EWOULDBLOCK) return {};
    else throw std::runtime_error(system_error("read offline jobs"));
  }
}

std::uint64_t Channel::event(std::string line) {
  line += '\n';
  if (line.size() > kMaxEventsBytes - output_bytes_)
    throw std::runtime_error("offline event output exceeds its 16 MiB bound");
  if (output_.empty()) progress_ = std::chrono::steady_clock::now();
  output_bytes_ += line.size();
  output_.push_back(std::move(line));
  return ++queued_events_;
}

void Channel::pump() {
  if (output_.empty() || !error_.empty()) return;
  // A decode step can emit one event per active request. Drain that batch
  // before the next step, rather than accumulating a one-line-per-poll tail.
  // Bound both bytes and writes so cancellation/input handling stays prompt.
  std::size_t budget = 64 * 1024;
  for (unsigned writes = 0; writes < 64 && budget && !output_.empty(); ++writes) {
    const auto& line = output_.front();
    const auto count = ::write(output_fd_, line.data() + output_offset_,
        std::min(budget, line.size() - output_offset_));
    if (count > 0) {
      output_offset_ += count;
      output_bytes_ -= count;
      budget -= count;
      progress_ = std::chrono::steady_clock::now();
      if (output_offset_ == line.size()) {
        output_.pop_front(); output_offset_ = 0; ++written_events_;
      }
    } else {
      if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        error_ = system_error("write offline events");
      break;
    }
  }
  if (std::chrono::steady_clock::now() - progress_ > kOutputTimeout)
    error_ = "offline event reader made no progress for 30 seconds";
}
}  // namespace gewell::app::offline
