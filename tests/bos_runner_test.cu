#include "../src/diagnostics/bos_runner.cu"
namespace gewell::bos_runner {
bool run_self_tests(std::string* failure) {
  try {
    if (!weight_ids_are_valid()) {
      fail("weight map", "compile-time role/layer map is invalid");
    }
    if (ScratchLayout::kBytes != 1'341'184) {
      fail("scratch layout", "byte count changed");
    }
    const std::vector<CaptureSpec> specs = make_capture_specs();
    if (specs.size() != kCaptureCount) {
      fail("capture inventory", "expected 114 tensors");
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      if (specs[index].elements == 0 ||
          (index != 0 && specs[index - 1].name >= specs[index].name)) {
        fail("capture inventory", "names are duplicated or not canonical");
      }
    }
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}
int main() {
  std::string failure;
  if (gewell::bos_runner::run_self_tests(&failure)) return 0;
  std::cerr << failure << '\n';
  return 1;
}
