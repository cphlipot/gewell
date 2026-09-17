#pragma once
#include "gewell/runtime/backend.h"
#include "gewell/nvfp4_policy.h"
namespace gewell::gemma4_31b::sm120 {
class WeightArena;
std::unique_ptr<runtime::ExecutionBackend> make_runtime_backend(
    const WeightArena& weights, const runtime::BatchLimits& limits,
    nvfp4::ActivationPolicy activation_policy);
}
