# Model bundles and conversion

A serving bundle contains text weights and tokenizer data:

```text
model-directory/
  manifest.json
  weights.gwt
  tokenizer.json
```

All main weight precisions use `.gwt`. The header and per-tensor metadata
identify storage; the filename does not select precision. Startup validates
weight structure and bundle integrity. `gewell verify WEIGHT_FILE` additionally
scans the full weight payload and padding. Checksums detect corrupted files;
repositories, commits, and source hashes are not model allowlists.

`tokenizer.json` supplies the vocabulary and BPE merges. Its token IDs and
pipeline must match the supported Gemma 4 contract. Tokenization behavior,
chat rendering, model architecture, and generation defaults are implemented
in the runtime. Source configuration and chat-template files are not bundled
or interpreted as runtime settings. Python reference and calibration tools
read those files from their original source checkpoints.

## Download a prepared bundle

Download a published bundle into a local directory:

```bash
MODEL_REPO=organization/model-bundle
MODEL_DIR=/absolute/path/to/model-bundle
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
```

Use `hf auth login` if the repo requires authentication. An optional
`--revision` selects a particular release for reproducibility. Once the
bundle is local, serving needs no Hugging Face connection.

## Convert safetensors

Conversion uses Python 3.10+ and NumPy. `tools/convert.py` accepts a Gemma 4 31B safetensors directory or
single file through `--snapshot`. Fine-tunes and repacked checkpoints are
accepted when required tensor names, shapes, and storage layouts match.
Shard names and counts are unrestricted. A safetensors index is optional;
when present, it must agree with the shard contents. Unrelated tensors are
allowed, and text conversion does not require vision or assistant weights.

Supported source encodings are BF16, FP16, FP32, E4M3 FP8 (with an optional
scalar FP32 `weight_scale`), and packed NVFP4: U8 pairs of E2M1 weights,
group-16 E4M3 `weight_scale`, and scalar FP32 `weight_scale_2`.
Other quantization layouts require decoding to one of these encodings first.
Packed projections also need an activation `input_scale` from the source or
`--input-scales`.

```bash
SOURCE_DIR=/absolute/path/to/gemma4-31b-safetensors
MODEL_DIR=/absolute/path/to/gewell-model

python3 tools/convert.py --snapshot "$SOURCE_DIR" --plan
python3 tools/convert.py --snapshot "$SOURCE_DIR" --output "$MODEL_DIR"
build/gewell verify "$MODEL_DIR/weights.gwt"
```

The tokenizer defaults to `tokenizer.json` in the source directory. Use
`--serving-snapshot DIR` when it is stored separately. Conversion keeps source files in place and
refuses to replace existing output weights or a manifest. `--plan` validates
metadata and scales without copying weights. BF16 text weights occupy about
57.2 GiB; packed projections reduce that size. Conversion streams tensors,
with temporary decoding storage bounded by one tensor.

## Select precision

A mask contains `LAYER PROJECTION STORAGE` rules. `LAYER` is `*` or `0..59`;
storage is `bf16`, `fp8_w8a8`, or `nvfp4_w4a4`. Projection names are `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Global
layers share K/V, so there is no separate global V projection.

For example:

```text
* gate_proj nvfp4_w4a4
* up_proj fp8_w8a8
* down_proj bf16
```

Later rules override earlier ones. Omitted projections retain their source
storage; FP16 and FP32 inputs become BF16. Embeddings and norms use BF16.
Newly quantized or requantized projections require calibrated activation
scales in a JSON object keyed by logical name, such as
`{"layers.0.gate_proj.weight": 0.125}`. That number illustrates syntax;
measure scales for the intended model and recipe.

```bash
python3 tools/convert.py --snapshot "$SOURCE_DIR" \
  --mask /path/to/recipe.mask --input-scales /path/to/scales.json \
  --output /path/to/converted-model
```

Native `.gwt` input is also supported:

```bash
python3 tools/convert.py --artifact "$MODEL_DIR/weights.gwt" \
  --mask /path/to/recipe.mask --input-scales /path/to/scales.json \
  --output /path/to/repacked-model
python3 tools/convert.py --verify /path/to/repacked-model/weights.gwt
```

A lower-precision source can be decoded to a higher-precision target. The
converter warns and proceeds: storing NVFP4 values in BF16 or FP8 cannot
recover information already lost during quantization. Unchanged native
entries preserve their packed bytes and scales.

## Optional assistant and vision

Load a structurally compatible BF16 assistant directly:

```bash
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 1 --kv-cache-gpu-mib 8192 \
  --assistant /path/to/assistant/model.safetensors --mtp-depth 3
```

Extract the vision tower and projector from any source containing the required
tensors, using the same safetensors reader and supported encodings as text
conversion:

```bash
python3 tools/extract_vision.py --snapshot "$SOURCE_DIR" \
  --output /path/to/vision.safetensors
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 1 --kv-cache-gpu-mib 8192 \
  --vision /path/to/vision.safetensors
```

Extraction preserves BF16 values and decodes other supported encodings to
BF16, warning when widening quantized weights. Vision extraction does not
require text weights. Runtime component files currently use BF16.

The two component paths can be combined. Omitting `--vision` rejects image
requests. Omitting `--assistant` with a positive depth warns and forces depth
zero. Depth zero never loads assistant weights.

Run each tool with `--help` for its complete argument list. Calibration reports
and recommended recipes belong with the published bundle.
