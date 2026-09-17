#pragma once

#include <cstdint>

namespace gewell::nvfp4 {

enum class ActivationPolicy : std::uint8_t { always, prefill };
enum class Phase : std::uint8_t { prefill, decode };

constexpr bool fp4_activations(ActivationPolicy policy, Phase phase) {
  return policy == ActivationPolicy::always || phase == Phase::prefill;
}

constexpr const char* activation_policy_name(ActivationPolicy policy) {
  return policy == ActivationPolicy::prefill ? "prefill" : "always";
}

}  // namespace gewell::nvfp4
