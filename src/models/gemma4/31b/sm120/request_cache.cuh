#pragma once

#include "execution_cache.h"

namespace gewell::gemma4_31b::sm120 {

class RuntimeKvCaches {
 public:
  explicit RuntimeKvCaches(std::uint32_t global_capacity,
                           ExecutionCache* persistent = nullptr,
                           kv_cache::ExecutionId execution = 0,
                           kv_cache::Format local = kv_cache::Format::bf16,
                           kv_cache::Format global = kv_cache::Format::bf16)
      : global_capacity_(global_capacity),
        persistent_(persistent),
        execution_(execution), local_format_(local), global_format_(global) {
    if (persistent_ == nullptr) {
      local_key_ = std::make_unique<DeviceAllocation>(local_bytes_per_kind());
      local_value_ =
          std::make_unique<DeviceAllocation>(local_bytes_per_kind());
      global_kv_ = std::make_unique<DeviceAllocation>(
          std::size_t(model::kGlobalLayerCount) * model::kGlobalKvHeadCount * global_capacity *
          kv_cache::row_bytes(640, global_format_, 2));
    } else if (execution_ == 0) {
      fail("generation KV cache", "persistent execution is missing");
    }
  }

  [[nodiscard]] LayerCacheView layer(std::uint32_t layer_index) const {
    if (layer_index >= model::kLayerCount) {
      fail("generation KV cache view", "layer is outside the model");
    }
    if (persistent_ != nullptr) {
      return persistent_->layer(execution_, layer_index);
    }
    if (model::is_global_layer(layer_index)) {
      const std::size_t slot = layer_index / 6;
      const std::size_t elements_per_layer =
          std::size_t(model::kGlobalKvHeadCount) * global_capacity_ * kv_cache::row_words(640, global_format_, 2);
      return {
          static_cast<BFloat16*>(global_kv_->data()) +
              slot * elements_per_layer,
          nullptr,
          global_capacity_,
          nullptr, nullptr, 0, 0, 0, 0, global_format_,
      };
    }
    const std::size_t slot = layer_index - layer_index / 6;
    const std::size_t elements_per_layer =
        static_cast<std::size_t>(model::kLocalKvHeadCount) *
        kLocalCacheCapacity * kv_cache::row_words(model::kLocalHeadSize, local_format_);
    return {
        static_cast<BFloat16*>(local_key_->data()) + slot * elements_per_layer,
        static_cast<BFloat16*>(local_value_->data()) +
            slot * elements_per_layer,
        kLocalCacheCapacity,
        nullptr, nullptr, 0, 0, 0, 0, local_format_,
    };
  }

  [[nodiscard]] std::size_t local_bytes() const {
    return persistent_ == nullptr
               ? local_key_->size() + local_value_->size()
               : persistent_->config().local_ring_bytes;
  }
  [[nodiscard]] std::size_t global_bytes() const {
    return persistent_ == nullptr ? global_kv_->size()
                                  : persistent_->config().global_page_bytes;
  }

 private:
  std::size_t local_bytes_per_kind() const {
    return std::size_t(model::kLocalLayerCount) * model::kLocalKvHeadCount * kLocalCacheCapacity *
        kv_cache::row_bytes(model::kLocalHeadSize, local_format_);
  }
  std::uint32_t global_capacity_{};
  ExecutionCache* persistent_{};
  kv_cache::ExecutionId execution_{};
  kv_cache::Format local_format_, global_format_;
  std::unique_ptr<DeviceAllocation> local_key_;
  std::unique_ptr<DeviceAllocation> local_value_;
  std::unique_ptr<DeviceAllocation> global_kv_;
};


}  // namespace gewell::gemma4_31b::sm120
