#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gewell::gemma4_31b {

enum class Component { assistant, vision };

struct ComponentTensor {
  std::size_t physical_id;
  const std::uint8_t* data;
  std::size_t bytes;
};

// Original safetensors names and shapes; GPU alignment is supplied at loading.
class ComponentFile {
 public:
  ComponentFile(const std::string& path, Component component);
  ~ComponentFile();
  ComponentFile(const ComponentFile&) = delete;
  ComponentFile& operator=(const ComponentFile&) = delete;
  const std::vector<ComponentTensor>& tensors() const { return tensors_; }
  std::size_t device_bytes() const { return device_bytes_; }
 private:
  const std::uint8_t* mapping_{};
  std::size_t mapping_bytes_{};
  std::size_t device_bytes_{};
  std::vector<ComponentTensor> tensors_;
};

std::string component_tensor_name(std::size_t physical_id);

}  // namespace gewell::gemma4_31b
