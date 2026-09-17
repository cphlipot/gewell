# Build

Build on Linux with an NVIDIA CUDA toolkit that supports `sm_120a`. The
development toolchain uses CUDA 13.1. The executable targets compute capability
12.0, including RTX PRO 6000 Blackwell and RTX 5090.

In theory, there is nothing that prevents you from building and running it on Windows, but it was not tested yet.

Required tools and libraries:

- CMake 3.25 or newer, a C++17 compiler supported by your CUDA toolkit, and Ninja
  or Make.
- CUDA development libraries, including cuBLAS and cuBLASLt, and a compatible
  NVIDIA driver for execution.
- OpenSSL, libpng, and libjpeg development packages, and the `patch` command.

On Debian/Ubuntu, the non-CUDA packages can be installed with:

```bash
sudo apt-get install build-essential cmake ninja-build patch libssl-dev libpng-dev libjpeg-dev
```

CMake downloads XGrammar and CUTLASS sources during
configuration. To use already downloaded
sources, set `FETCHCONTENT_SOURCE_DIR_XGRAMMAR` and
`FETCHCONTENT_SOURCE_DIR_CUTLASS` to their source directories. The XGrammar
directory must already have `vendor/xgrammar/unicode-escapes.patch` applied
when bypassing the download step.

## Engine

Run from the source checkout:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DGEWELL_BUILD_DIAGNOSTICS=OFF
cmake --build build --target gewell --parallel 4
build/gewell --help
```

If CUDA is outside the compiler's search path, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` at configuration time. Lower the
parallel job count if compilation exhausts host memory.

Optionally install the executable into a chosen prefix:

```bash
cmake --install build --prefix "$HOME/.local"
```

The executable dynamically links CUDA and native system libraries. Continue with
[model preparation](models.md) and [server launch](launch.md).

## Tests

The CPU libraries and their tests can be built without CUDA or model weights:

```bash
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGEWELL_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build/host --parallel 4
ctest --test-dir build/host --output-on-failure
```

The tokenizer, grammar, and HTTP API tests skip when their reference tokenizer is
absent. To run them, add `-DGEWELL_TOKENIZER_DIRECTORY=/path/to/serving-snapshot`
using `tokenizer.json` for the reference fixtures.
CPU mode builds libraries and tests; it does not build a CPU inference engine.

For CUDA tests, configure the main build with `-DBUILD_TESTING=ON`, build it,
then run `ctest --test-dir build --output-on-failure` on the supported GPU.
`-DGEWELL_BUILD_DIAGNOSTICS=ON` additionally builds `gewell_diagnostics` for
fixed oracle captures and profiling; those commands need their own fixtures.

The retained Python artifact tests use Python 3.10+ and NumPy:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pytest
.venv/bin/python -B -m pytest tests/test_*.py
```

These Python tests exercise conversion, packing, corruption rejection, and
structural compatibility with small fixtures. They do not load a full model on a GPU.
The optional real-asset copy test skips unless `GEWELL_TEST_SERVING_SNAPSHOT`
points to a local serving directory; set it to include that check.
