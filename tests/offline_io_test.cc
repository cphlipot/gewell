#include "app/offline_io.h"
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace gewell::app::offline;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Temporary {
  char pattern[64]{"/tmp/gewell-offline-io-XXXXXX"};
  std::filesystem::path path;
  Temporary() : path(::mkdtemp(pattern)) {}
  ~Temporary() { std::filesystem::remove_all(path); }
};

void regular_outputs(const std::filesystem::path& directory) {
  const auto path = (directory / "logits.bf16").string();
  {
    Output output(path, true);
    output.append("abc", 3);
    output.pump();
    require(output.ready(), "regular output did not flush");
    bool rejected = false;
    try { Output duplicate(path, true); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "existing output was overwritten");
    output.complete();
  }
  std::ifstream input(path);
  std::string bytes;
  input >> bytes;
  require(bytes == "abc", "regular output bytes changed");
  const auto failed = (directory / "partial.bf16").string();
  { Output output(failed, false); output.append("partial", 7); output.pump(); }
  require(!std::filesystem::exists(failed), "partial output survived failure");
  { Output output(failed, false); output.complete(); output.discard(); }
  require(!std::filesystem::exists(failed), "multi-output failure preserved an already closed partial result");
}

void fifo_backpressure(const std::filesystem::path& directory) {
  const auto path = (directory / "logits.fifo").string();
  require(::mkfifo(path.c_str(), 0600) == 0, "cannot create FIFO");
  Output output(path, true);
  require(!output.ready(), "FIFO opened without a reader");
  const int reader = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
  require(reader >= 0, "cannot open FIFO reader");
  output.pump();
  require(output.ready(), "FIFO did not connect to reader");
  std::vector<std::uint8_t> data(1024 * 1024, 0x5a);
  output.append(data.data(), data.size());
  output.pump();
  require(!output.ready(), "slow reader unexpectedly accepted a full logit chunk");
  // A blocked FIFO must not prevent unrelated local output or cancellation.
  const auto text = (directory / "other.u32").string();
  { Output other(text, false); other.append("ok", 2); other.pump(); other.complete(); }
  output.cancel();
  output.pump();
  require(!output.ready(), "cancelled FIFO was reopened");
  ::close(reader);
  require(std::filesystem::exists(path), "caller-owned FIFO was removed");

  const int disconnected = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
  Output broken(path, true);
  ::close(disconnected);
  broken.append("x", 1);
  broken.pump();
  require(!broken.error().empty(), "disconnected FIFO did not report EPIPE");
}

void bounded_channel(const std::filesystem::path& directory) {
  const auto input_path = directory / "jobs.jsonl";
  { std::ofstream file(input_path); file << "one\ntwo"; }
  const int input = ::open(input_path.c_str(), O_RDONLY);
  int pipe[2];
  require(::pipe(pipe) == 0, "cannot create output pipe");
  const auto flags = ::fcntl(pipe[1], F_GETFL);
  {
    Channel channel(input, pipe[1]);
    require(channel.read_line() == "one" && channel.read_line() == "two" &&
        !channel.read_line() && channel.eof(), "JSONL EOF did not preserve the last line");
    const auto first = channel.event("result");
    require(channel.written_events() < first, "event considered delivered before output");
    channel.pump();
    require(channel.written_events() == first, "event completion was not observed");
    channel.event(std::string(kMaxEventsBytes - 1, 'x'));
    bool rejected = false;
    try { channel.event("overflow"); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && !channel.ready(), "event queue exceeded its bound");
    channel.pump();
    require(!channel.empty(), "slow event reader did not apply backpressure");
  }
  require(::fcntl(pipe[1], F_GETFL) == flags, "descriptor flags were not restored");
  ::close(input); ::close(pipe[0]); ::close(pipe[1]);
  { std::ofstream file(input_path); file << std::string(kMaxLineBytes + 1, 'x'); }
  const int too_long = ::open(input_path.c_str(), O_RDONLY);
  const int sink = ::open("/dev/null", O_WRONLY);
  {
    Channel channel(too_long, sink);
    bool rejected = false;
    try { channel.read_line(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "oversized JSONL line was accepted");
  }
  ::close(too_long); ::close(sink);
}

void batched_events() {
  const int input = ::open("/dev/null", O_RDONLY);
  int pipe[2];
  require(::pipe(pipe) == 0, "cannot create batched event pipe");
  require(::fcntl(pipe[0], F_SETFL, O_NONBLOCK) == 0, "cannot make event reader nonblocking");
  {
    Channel channel(input, pipe[1]);
    std::string expected;
    for (unsigned i = 0; i < 32; ++i) {
      const auto line = "token-" + std::to_string(i);
      expected += line + '\n';
      channel.event(line);
    }
    channel.pump();
    require(channel.empty() && channel.written_events() == 32,
        "one decode batch was not delivered in one poll");
    char buffer[4096];
    const auto count = ::read(pipe[0], buffer, sizeof(buffer));
    require(count == static_cast<ssize_t>(expected.size()) &&
        std::string(buffer, count) == expected, "batch event bytes/order changed");

    // Exercise partial writes, an output-full retry, and multiple lines after
    // the partial line. Delivery counters must advance only for full lines.
    const std::string large(256 * 1024, 'x');
    expected = large + "\nafter\n";
    channel.event(large);
    channel.event("after");
    channel.pump();
    require(!channel.empty() && channel.written_events() == 32,
        "partial event was marked delivered");
    channel.pump();  // Full pipe: return promptly without dropping pending bytes.
    require(channel.error().empty(), "ordinary backpressure became an output error");
    std::string actual;
    for (unsigned attempt = 0; attempt < 1024; ++attempt) {
      while (true) {
        const auto n = ::read(pipe[0], buffer, sizeof(buffer));
        if (n <= 0) break;
        actual.append(buffer, n);
      }
      if (channel.empty()) break;
      channel.pump();
    }
    require(channel.empty() && channel.written_events() == 34 && actual == expected,
        "partial batched event flush lost ordering or delivery accounting");
  }
  ::close(input); ::close(pipe[0]); ::close(pipe[1]);

  // A writable regular sink must still yield after a bounded output quantum.
  const int source = ::open("/dev/null", O_RDONLY), sink = ::open("/dev/null", O_WRONLY);
  {
    Channel channel(source, sink);
    for (unsigned i = 0; i < 128; ++i) channel.event("x");
    channel.pump();
    require(!channel.empty() && channel.written_events() == 64,
        "event pumping did not yield after its write quantum");
    channel.pump();
    require(channel.empty() && channel.written_events() == 128,
        "second event quantum did not finish");
    channel.event(std::string(128 * 1024, 'x'));
    channel.pump();
    require(!channel.empty() && channel.written_events() == 128,
        "event pumping did not yield after its byte quantum");
  }
  ::close(source); ::close(sink);
}
}  // namespace

int main() {
  try {
    std::signal(SIGPIPE, SIG_IGN);
    Temporary directory;
    regular_outputs(directory.path);
    fifo_backpressure(directory.path);
    bounded_channel(directory.path);
    batched_events();
    std::cout << "offline I/O tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
