// Benchmark-only selected-row capture. Uses the production prefill/weight
// paths with BF16 KV and attention; writes bounded BF16 tiles to a parent pipe.
#include "models/gemma4/31b/sm120/executor.cuh"
#include "models/gemma4/31b/sm120/runner_support.cuh"

#include <cerrno>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace sm = gewell::gemma4_31b::sm120;
namespace model = gewell::gemma4_31b;
using namespace sm;

namespace {
std::vector<std::uint32_t> read_u32(const std::string& path) {
  const auto bytes = std::filesystem::file_size(path);
  if (!bytes || bytes % 4 || bytes > 262144 * 4)
    throw std::runtime_error("invalid u32 file size: " + path);
  std::vector<std::uint32_t> data(bytes / 4);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("short or changed u32 file: " + path);
  return data;
}

void write_pipe(int fd, const void* data, std::size_t bytes) {
  auto* cursor = static_cast<const char*>(data);
  while (bytes) {
    const auto count = ::write(fd, cursor, bytes);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("selected-logit pipe write failed");
    cursor += count;
    bytes -= count;
  }
}

struct Request {
  std::string id;
  std::vector<std::uint32_t> tokens, positions;
};

std::vector<Request> read_requests(const std::string& path) {
  if (std::filesystem::file_size(path) > 8 * 1024 * 1024)
    throw std::runtime_error("request manifest too large");
  std::ifstream input(path);
  std::vector<Request> result;
  std::set<std::string> ids;
  std::size_t total_tokens = 0;
  std::string line;
  while (std::getline(input, line)) {
    const auto first = line.find('\t'), second = line.find('\t', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        line.find('\t', second + 1) != std::string::npos)
      throw std::runtime_error("expected id, tokens.u32, positions.u32");
    Request r{line.substr(0, first), read_u32(line.substr(first + 1, second - first - 1)),
              read_u32(line.substr(second + 1))};
    if (r.id.empty() || r.id.find_first_not_of("0123456789abcdef") != std::string::npos ||
        !ids.insert(r.id).second || result.size() == 4096 || r.positions.size() > 128 ||
        r.tokens.size() >= 262144)
      throw std::runtime_error("invalid or duplicate request");
    if (*std::max_element(r.tokens.begin(), r.tokens.end()) >= model::kVocabSize ||
        r.positions.back() >= r.tokens.size() || !std::is_sorted(r.positions.begin(), r.positions.end()) ||
        std::adjacent_find(r.positions.begin(), r.positions.end()) != r.positions.end())
      throw std::runtime_error("invalid token or selected position");
    total_tokens += r.tokens.size();
    if (total_tokens > 64 * 1024 * 1024)
      throw std::runtime_error("request token payload exceeds 256 MiB");
    result.push_back(std::move(r));
  }
  if (!input.eof() || result.empty()) throw std::runtime_error("invalid request manifest");
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: teacher_replay ARTIFACT REQUESTS.tsv LOGITS_FD");
    std::size_t parsed = 0;
    const auto fd = std::stoi(argv[3], &parsed);
    struct stat info{};
    if (parsed != std::strlen(argv[3]) || fd <= 2 || ::fstat(fd, &info) || !S_ISFIFO(info.st_mode))
      throw std::runtime_error("LOGITS_FD must be an inherited pipe");
    const auto requests = read_requests(argv[2]);
    gewell::console::set_format("json");
    validate_cuda_device();
    auto artifact = gewell::artifact::ArtifactFile::Open(argv[1]);
    WeightArena weights(artifact);
    constexpr std::uint32_t chunk_rows = 1024, head_rows = 32;
    const auto tile_bytes = std::size_t(head_rows) * model::kVocabSize * sizeof(BFloat16);
    DeviceAllocation logits(tile_bytes), capped(tile_bytes);
    PinnedHostAllocation host(tile_bytes);
    std::cout << "{\"event\":\"teacher_replay_identity\",\"kv\":\"bf16\","
                 "\"attention_compute\":\"bf16\",\"mtp\":0,\"chunk_rows\":1024,"
                 "\"head_rows\":32,\"native_nvfp4\":" << (weights.has_native() ? "true" : "false")
              << ",\"native_fp8\":" << (weights.has_fp8() ? "true" : "false") << "}\n" << std::flush;
    for (const auto& request : requests) {
      const auto start = std::chrono::steady_clock::now();
      Executor engine(weights, request.tokens.size(), 1, false, {}, nullptr, 0, {}, 0, {}, chunk_rows,
          gewell::nvfp4::ActivationPolicy::always, 1, gewell::kv_cache::Format::bf16,
          gewell::kv_cache::Format::bf16, gewell::attention::Compute::bf16, gewell::attention::Compute::bf16);
      std::size_t selected = 0;
      for (std::uint32_t base = 0; base < request.tokens.size(); base += chunk_rows) {
        const auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(chunk_rows, request.tokens.size() - base));
        engine.replay_chunk(request.tokens.data() + base, base, rows);
        while (selected < request.positions.size() && request.positions[selected] < base + rows) {
          const auto first = request.positions[selected];
          std::uint32_t count = 1;
          while (count < head_rows && selected + count < request.positions.size() &&
                 request.positions[selected + count] == first + count && first + count < base + rows)
            ++count;
          engine.replay_head(first - base, count, static_cast<BFloat16*>(logits.data()),
                             static_cast<BFloat16*>(capped.data()));
          const auto bytes = std::size_t(count) * model::kVocabSize * sizeof(BFloat16);
          check_cuda(cudaMemcpyAsync(host.data(), capped.data(), bytes, cudaMemcpyDeviceToHost, engine.stream()),
                     "copy selected teacher logits");
          check_cuda(cudaStreamSynchronize(engine.stream()), "finish selected teacher logits");
          write_pipe(fd, host.data(), bytes);
          selected += count;
        }
      }
      check_cuda(cudaStreamSynchronize(engine.stream()), "finish teacher replay");
      if (selected != request.positions.size()) throw std::runtime_error("selected row accounting differs");
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::cout << "{\"event\":\"teacher_replay_record\",\"key\":\"" << request.id
                << "\",\"input_tokens\":" << request.tokens.size() << ",\"rows\":" << selected
                << ",\"wall_seconds\":" << elapsed << "}\n" << std::flush;
    }
    ::close(fd);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "teacher replay: " << e.what() << '\n';
    return 1;
  }
}
