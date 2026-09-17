#pragma once
#include <string>
namespace gewell::diagnostics {
// Runs the pinned two-token Gemma 4 31B BF16 graph for token IDs [2, 902].
// The artifact metadata is validated before any weights are copied to CUDA.
// Use `gewell verify PATH` for the opt-in full payload check.
// A capture directory of "-" disables captures; any other path must not exist.
int run(const std::string& artifact_path, const std::string& capture_directory);

// Executes the same two-token case as two cache-backed M=1 decode steps. The
// local cache is a real 1,024-slot ring; the full-attention cache has the fixed
// two-token capacity needed by this slice. Capture output matches `pair`.
int run_cached(const std::string& artifact_path,
               const std::string& capture_directory);

// Runs the pinned 16-token prompt and eight-decision ordinary-cache decode.
// The prompt is ingested as fixed M=1 steps and is checked against the matching
// eager-BF16 sequential-M1 continuation; local caches retain the full 1,024-
// token ring and global caches retain the fixed 24-token horizon.
int run_short_decode(const std::string& artifact_path,
                     const std::string& capture_directory);

// Runs the pinned 1,026-token local-window boundary fixture as sequential M=1
// cache updates and makes one decision after absolute position 1,025. Local
// caches are fixed 1,024-slot rings; global caches retain all 1,026 positions.
int run_local_boundary(const std::string& artifact_path,
                       const std::string& capture_directory);

// Runs the same fixture with one fixed M=1024 prefill for positions 0..1023,
// then fixed M=1 updates at positions 1024 and 1025. The three predictions are
// pinned independently and captures use the canonical boundary.hybrid source.
int run_local_boundary_prefill(const std::string& artifact_path,
                               const std::string& capture_directory);

// Runs the fixed 1,024-token prefill followed by exactly 511 greedy M=1
// decode steps. The decode body is captured once and replayed on one explicit
// nonblocking stream; the resulting 512 uint32 token IDs can be written to an
// otherwise-new output directory.
int run_graph_decode(const std::string& artifact_path,
                     const std::string& output_directory);

// Builds the same fixed graph-decode engine, seeds it with a fresh 1,024-token
// prefill, and brackets exactly the position-1,024 graph replay with the CUDA
// profiler API. This is the short full-model entry point for Nsight captures.
int run_profile_decode(const std::string& artifact_path);

}  // namespace gewell::diagnostics
