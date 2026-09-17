# Offline generation

Use HTTP for text/chat tokenization and ordinary image input. These commands
accept prepared token or tensor files for batch jobs and integrations that own
prompt preparation.

## Token files

`PROMPT.u32` is a nonempty stream of little-endian uint32 token IDs in
`0..262143`. Output token files use the same encoding and exclude the prompt.
The processed horizon must satisfy `prompt_tokens + new_tokens - 1 <= 262144`
and fit the selected memory configuration.

```bash
build/gewell generate ARTIFACT PROMPT.u32 128 OUTPUT.u32
```

`generate` defaults to greedy sampling and does not stop at EOS. It writes
exactly the requested count. Output paths must be new. Optional `LOGITS.bf16`
contains post-softcap rows shaped `[new_tokens, 262144]`, occupying 524,288
bytes per output decision.

## TSV batches

```bash
build/gewell generate-batch ARTIFACT requests.tsv 4 8192 output-directory
```

Each line contains nine tab-separated fields, without a header:

```text
id  prompt_file  max_new_tokens  temperature  top_p  top_k  seed  honor_eos  capture_logits
```

The spaces above show columns; the actual file uses tabs. IDs are unique,
at most 128 characters, and match `[A-Za-z0-9_-][A-Za-z0-9_.-]*`. Relative
prompt paths resolve against the TSV directory. `honor_eos` and
`capture_logits` are `0` or `1`; enabled EOS stopping includes the selected
stop token in the output file. Sampling values follow the [CLI rules](cli.md).

The output directory must not exist. Successful requests create their token
files and optional logits; the run records per-request accounting and a summary.
Batch capacity limits active work, and finished requests are replaced from the
queue while the model remains resident.

## Resident JSONL jobs

```bash
build/gewell run-jobs ARTIFACT 4 8192 --kv-cache-cpu-mib 1024
```

Each input line is one JSON object. IDs must be nonempty, at most 256 UTF-8
bytes, and distinct while active. Paths are local to the server process.

```json
{"op":"generate","id":"sample-1","prompt_path":"/tmp/prompt.u32","max_tokens":32,"temperature":0.8,"top_p":0.95,"seed":73,"honor_eos":true,"outputs":{"tokens":"/tmp/sample-1.u32"}}
{"op":"prefill","id":"warm","prompt_path":"/tmp/prompt.u32","cache":{"prompt_id":"document"},"checkpoint_offsets":[17,33]}
{"op":"finish","id":"release","prompt_id":"document"}
{"op":"stats","id":"inventory"}
{"op":"cancel","id":"sample-1"}
```

`honor_eos` defaults to true; false emits exactly `max_tokens`. Temperature
defaults to zero, top-p to one, and top-k to zero. Output paths must be new.
`outputs.logits` optionally names a BF16 logit file. `cache` accepts the same
retention controls described in the [cache guide](cache.md).

Stdout emits `offline_ready`, `offline_tokens`, and a terminal `offline_result`,
`offline_cancelled`, or `offline_error` event for each generation. Diagnostics
go to stderr. Completed output files precede success events; failed or
cancelled requests remove partial regular outputs. EOF drains accepted work;
SIGINT/SIGTERM cancels pending work and cleans up.

## Prepared images

Start `run-jobs` with `--vision VISION.safetensors` to allow generation and
prefill requests to include `prepared_images`:

```json
{"prepared_images":[{"pixel_values":"/tmp/image.pixels.f32","position_ids":"/tmp/image.positions.i32","max_soft_tokens":280}]}
```

Each image budget is 70, 140, 280, 560, or 1120 soft tokens. Tensor files must
match the selected budget, and every image must match an image-placeholder
span in the prompt. List images in prompt order. Image spans stay complete
during prefill, while surrounding text is chunked normally.

`gewell --vision VISION.safetensors caption ARTIFACT PROMPT.u32 PIXELS.f32 POSITIONS.i32 MAX_NEW_TOKENS OUTPUT.u32
[LOGITS.bf16]` is the single-image variant. It stops on EOS and includes the
stop token in the output. Preparing these tensors requires the model's
image processing contract; HTTP accepts PNG/JPEG directly.
