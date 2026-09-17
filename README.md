# Gewell

Gewell is a single-GPU inference engine for Gemma 4 31B (more to come) on NVIDIA Blackwell
GPUs with compute capability 12.0 (`sm_120a`) (*potentially* more to come). It serves text and image chat
through an OpenAI-compatible HTTP API and supports local token-based generation.

Features include streamed responses, tool conversations, JSON-constrained
answers, continuous batching, prefix caching with optional CPU storage, and
MTP speculative decoding using the Gemma assistant. Local and global
KV storage can independently use BF16 or FP8.

The project is pre-release: artifact formats and CLI contracts can change.

## Get started

1. [Build the engine](docs/build.md).
2. [Obtain or convert a model bundle](docs/models.md).
3. [Launch the server and send a request](docs/launch.md).


## Reference

- [Command-line interface](docs/cli.md)
- [HTTP API](docs/http-api.md)
- [Prefix caching, prefill, and retention](docs/cache.md)
- [Offline jobs and token files](docs/offline.md)
- [HTTP benchmarking](docs/benchmark.md)

This repository contains engine sources, tests, documentation, and preparation
tools. Model weights are distributed separately in Hugging Face model repos.
Use the complete bundle and revision specified by its model card.
