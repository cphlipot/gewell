# HTTP benchmark tool

`tools/http-bench` builds a standalone client that measures an already running
HTTP server. It needs CMake 3.25+, a C++17 compiler, libcurl 7.85+ development
files, and OpenSSL development files. It does not need CUDA or model weights.

```bash
cmake -S tools/http-bench -B build/http-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build/http-bench --parallel 4
build/http-bench/gewell-http-bench \
  --base-url http://127.0.0.1:6311/v1 --model gemma-4-31b \
  --scenarios tools/http-bench/scenarios.json --scenario chat-mixed \
  --output /tmp/gewell-http-measurement
```

The output directory must not exist. `--plan` validates the scenario and shows
the selected matrix without opening connections or writing results. Omit
`--scenario` to run every scenario. The bundled scenarios exercise the client;
measure real deployments with representative prompts and output lengths.

The client records per-request timing and results, including streaming latency
and throughput. Compare runs that share the artifact, settings, prompts,
concurrency, and measurement boundaries; server-side token rates exclude
client and queueing time.

`--help` lists timeout, output-limit, and authentication options. When measuring
a proxy that requires authentication, `--api-key-env VARIABLE` or
`--api-key-file PATH` supplies a key without writing it to the result files.

Its protocol tests use a local fixture server and Python:

```bash
cmake -S tools/http-bench -B build/http-bench -DBUILD_TESTING=ON
cmake --build build/http-bench --parallel 4
ctest --test-dir build/http-bench --output-on-failure
```
