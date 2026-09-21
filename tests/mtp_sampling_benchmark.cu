// Fixed-input, eager-launch sampler timings. Emits raw CUDA-event samples as
// JSON lines; this excludes the assistant/target forward passes and KV commit.
#include "gewell/mtp_sampling.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace m = gewell::mtp_sampling;
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T> struct Device {
  T* data{};
  explicit Device(std::size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
  ~Device() { cudaFree(data); }
  void put(const std::vector<T>& values) {
    check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

std::vector<float> measure(const std::function<void()>& run) {
  constexpr unsigned warmups = 10, samples = 11, iterations = 100;
  for (unsigned i = 0; i < warmups; ++i) run();
  check(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  std::vector<float> times;
  for (unsigned sample = 0; sample < samples; ++sample) {
    check(cudaEventRecord(begin));
    for (unsigned i = 0; i < iterations; ++i) run();
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms, begin, end));
    times.push_back(ms * 1000 / iterations);
  }
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  return times;
}

void benchmark(unsigned k, bool compact, bool reject) {
  constexpr unsigned vocabulary = 262144, depth = 3;
  const unsigned size = compact ? k : vocabulary;
  Device<__nv_bfloat16> draft_logits(vocabulary), target_logits(vocabulary);
  Device<float> dense((2 * depth + 1) * vocabulary), draws(2 * depth + 1);
  Device<m::TokenProbability> sparse((2 * depth + 1) * std::max(k, 1U));
  Device<unsigned> ids(depth), outputs(depth + 1);
  Device<m::Result> result(1);
  Device<m::Status> status(1);
  const auto bytes = m::scratch_bytes(vocabulary);
  Device<unsigned char> scratch(bytes);
  std::vector<__nv_bfloat16> q(vocabulary), p(vocabulary);
  for (unsigned token = 0; token < vocabulary; ++token) {
    q[token] = __float2bfloat16(float(int((token * 2654435761U) % 65521) - 32760) / 4096);
    p[token] = q[token];
  }
  if (reject) std::rotate(p.begin(), p.begin() + 1337, p.end());
  draft_logits.put(q); target_logits.put(p);
  draws.put({0.371F, 0.673F, 0.129F, 0.99F, 0.99F, 0.99F, 0.751F});
  m::clear_status(status.data);
  const auto build = [&](bool target, unsigned row) {
    const auto* logits = target ? target_logits.data : draft_logits.data;
    const unsigned offset = (target ? depth : 0) + row;
    if (compact)
      m::build_compact_distribution(logits, vocabulary, 0.8F, 0.95F, size,
          sparse.data + offset * size, scratch.data, bytes, status.data);
    else
      m::build_distribution(logits, vocabulary, 0.8F, 0.95F, k,
          dense.data + offset * size, scratch.data, bytes, status.data);
  };
  const auto sample = [&](unsigned row) {
    if (compact)
      m::sample_compact_distribution(sparse.data + row * size, size, vocabulary,
          draws.data + row, ids.data + row, status.data);
    else
      m::sample_distribution(dense.data + row * size, vocabulary,
          draws.data + row, ids.data + row, scratch.data, bytes, status.data);
  };
  const auto verify = [&] {
    if (compact)
      m::verify_compact_sequence(sparse.data + depth * size, sparse.data,
          ids.data, depth, size, vocabulary, draws.data + depth,
          draws.data + 2 * depth, outputs.data, result.data, status.data);
    else
      m::verify_sequence(dense.data + depth * size, dense.data, ids.data, depth,
          vocabulary, draws.data + depth, draws.data + 2 * depth, outputs.data,
          result.data, scratch.data, bytes, status.data);
  };
  const auto cycle = [&] {
    for (unsigned row = 0; row < depth; ++row) { build(false, row); sample(row); }
    for (unsigned row = 0; row <= depth; ++row) build(true, row);
    verify();
  };
  cycle();
  for (unsigned stage = 0; stage < 4; ++stage) {
    const std::function<void()> operation = stage == 0 ? std::function<void()>([&] { build(false, 0); })
        : stage == 1 ? std::function<void()>([&] { sample(0); })
        : stage == 2 ? std::function<void()>(verify) : std::function<void()>(cycle);
    const auto samples = measure(operation);
    m::Status observed;
    check(cudaMemcpy(&observed, status.data, sizeof(observed), cudaMemcpyDeviceToHost));
    if (observed != m::Status::success) throw std::runtime_error(m::status_message(observed));
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "{\"kind\":\"sampler\",\"path\":\"" << (compact ? "compact" : "dense")
              << "\",\"top_k\":" << k << ",\"top_p\":0.95,\"temperature\":0.8,\"depth\":3"
              << ",\"reject_fixture\":" << (reject ? "true" : "false")
              << ",\"stage\":\"" << std::vector<const char*>{"build_row", "sample_row", "verify", "cycle"}[stage]
              << "\",\"warmups\":10,\"iterations_per_sample\":100,\"median_us\":" << sorted[sorted.size()/2]
              << ",\"active_probability_bytes\":" << (2 * depth + 1) * size * (compact ? 8 : 4)
              << ",\"scratch_bytes\":" << bytes << ",\"samples_us\":[";
    for (unsigned i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
    std::cout << "]}" << std::endl;
  }
}

void benchmark_compact_batch(unsigned batch, unsigned k) {
  constexpr unsigned vocabulary = 262144;
  const auto bytes = m::scratch_bytes(vocabulary);
  Device<__nv_bfloat16> logits(std::size_t(batch) * vocabulary);
  Device<m::TokenProbability> output(batch * k), reference(batch * k);
  Device<m::Status> status(batch);
  Device<unsigned char> scratch(batch * bytes);
  std::vector<__nv_bfloat16> values(std::size_t(batch) * vocabulary);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = __float2bfloat16(float(int((i * 2654435761U) % 65521) - 32760) / 4096);
  logits.put(values);
  status.put(std::vector<m::Status>(batch, m::Status::success));
  std::vector<m::CompactDistributionInput> inputs;
  for (unsigned i = 0; i < batch; ++i) {
    inputs.push_back({logits.data + std::size_t(i) * vocabulary, 0.8F, 0.95F, k,
        output.data + i * k, scratch.data + i * bytes, bytes, status.data + i, nullptr});
    m::build_compact_distribution(inputs.back().logits, vocabulary, 0.8F, 0.95F, k,
        reference.data + i * k, inputs.back().scratch, bytes, inputs.back().status);
  }
  for (bool batched : {false, true}) {
    const auto operation = [&] {
      if (batched) m::build_compact_distributions(inputs, vocabulary);
      else for (const auto& row : inputs)
        m::build_compact_distribution(row.logits, vocabulary, row.temperature,
            row.top_p, k, row.output, row.scratch, bytes, row.status);
    };
    auto samples = measure(operation);
    std::vector<m::TokenProbability> expected(batch * k), actual(batch * k);
    check(cudaMemcpy(expected.data(), reference.data, expected.size() * sizeof(expected[0]), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(expected[0])))
      throw std::runtime_error("batched compact probabilities differ from serial reference");
    std::vector<m::Status> statuses(batch);
    check(cudaMemcpy(statuses.data(), status.data, batch * sizeof(statuses[0]), cudaMemcpyDeviceToHost));
    if (std::any_of(statuses.begin(), statuses.end(), [](auto s) { return s != m::Status::success; }))
      throw std::runtime_error("batched compact sampler returned failure");
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "{\"kind\":\"compact_batch\",\"batch\":" << batch << ",\"top_k\":" << k
              << ",\"batched\":" << (batched ? "true" : "false")
              << ",\"exact\":true,\"median_us\":" << sorted[sorted.size()/2] << ",\"samples_us\":[";
    for (unsigned i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
    std::cout << "]}" << std::endl;
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, 0));
    int driver, runtime;
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::cout << "{\"kind\":\"machine\",\"gpu\":\"" << properties.name
              << "\",\"driver_api\":" << driver << ",\"runtime\":" << runtime << "}" << std::endl;
    if (argc == 2 && std::strcmp(argv[1], "--compact-batch") == 0) {
      for (unsigned batch : {1U, 4U, 32U, 64U})
        for (unsigned k : {40U, 256U}) benchmark_compact_batch(batch, k);
      return 0;
    }
    if (argc != 1) throw std::runtime_error("expected no arguments or --compact-batch");
    for (const unsigned k : {40U, 64U, 256U}) {
      benchmark(k, false, false);
      benchmark(k, true, false);
    }
    benchmark(40, false, true);
    benchmark(40, true, true);
    benchmark(0, false, false);
    benchmark(1024, false, false);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
