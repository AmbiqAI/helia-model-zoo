# Add a model to the heliaAOT release corpus

The model zoo owns artifact identity and integrity. It does not decide which
models helia-aot runs as release coverage. A model becomes eligible for release
testing only after its model, golden fixture, model card, and record
merge here; a separate helia-aot change then pins that merged commit and adds a
release case.

## 1. Create a self-contained model directory

Choose a stable, descriptive ID (lowercase letters, digits, `.` and `-`). IDs
are an API consumed by helia-aot and other tools and should not be renamed.
Place the files under `models/<id>/`:

```text
models/<id>/
  record.json
  README.md
  <precision>/model.tflite
  <precision>/golden.npz
```

Precisions are `fp32`, `fp16`, `a8w8`, `a16w8` and `a8w4`.

The repository's `.gitattributes` tracks `.tflite` and `.npz` files with Git
LFS. Confirm that Git sees new artifacts as LFS objects before committing:

```bash
git check-attr filter -- path/to/model.tflite path/to/golden.npz
git lfs status
```

`helia-zoo list` shows the new model once its record exists; there is no other
inventory to edit.

## 2. Write the model card

The model README is the provenance and license reference used by the record.
Document:

- what the model does and its input/output contract;
- where the model came from, including an upstream repository, release, or
  paper reference when available;
- the applicable upstream license and a link to its authoritative text; and
- any conversion, quantization, or other transformation applied to the
  checked-in artifact.

Record these in the card's `Source`, `License`, `Training` and
`Conversion` rows, pinning the upstream revision. Mark anything you cannot
confirm `TODO(verify)` rather than guessing. Add the model to [NOTICE](../../NOTICE),
and add its license text under `licenses/` when that license requires it.

Referencing an upstream license records the terms under which the third-party
model is distributed; it does not imply that Ambiq created or relicensed it.

## 3. Create the golden fixture

Release goldens must come from the pinned reference runtime, never from
helia-aot. For the standard deterministic generator:

```bash
python -m venv .golden-venv
. .golden-venv/bin/activate
python -m pip install -r tools/golden-requirements.txt
python -m pip install --no-deps -e .
helia-zoo golden generate path/to/model.tflite path/to/golden.npz --seed 42
```

The NPZ must contain consecutive `input_N` and `output_N` arrays whose shapes
and dtypes match the TFLite signature. The generator uses LiteRT's reference
kernels (`builtin_ref`); note that resolver in the record. For a streaming
model, prefer a `sequence` golden made from real input
(`helia-zoo golden generate --kind sequence --data ...`; see the README). If the
standard generator is unsuitable, document the deterministic input-generation
method and runtime version in the pull request and model card.

## 4. Add the record

helia-aot still reads `corpus-manifest-v1.json`, so a release model also needs a
v1 entry there, pointing at the same files:

```json
{
  "id": "example-int8",
  "model": "models/example/a8w8/model.tflite",
  "model_sha256": "<hydrated model SHA-256>",
  "golden": "models/example/a8w8/golden.npz",
  "golden_sha256": "<hydrated golden SHA-256>",
  "reference_runtime": "ai-edge-litert",
  "reference_runtime_version": "2.1.2",
  "provenance_reference": "models/example/README.md",
  "license_reference": "models/example/README.md"
}
```

Write `models/<id>/record.json` (`helia-model-zoo/record@1`), copying an
existing record as a template. Each file is a `path` inside the model directory
(relative, with no empty part and no part that starts with `.`) or a pinned
`hf://` URI. Take each tensor's name, shape, dtype, float scale and integer zero
point from the model, and record the LiteRT resolver its golden replays
under (`builtin_ref` for new goldens). CI refuses a v1 entry that disagrees with
the record holding its model, or whose ID is not that record's ID with a suffix
such as `-int8`.

The record's fields:

- `id` equals the directory name; `visibility` must be `public`, because this
  repository is public and CI refuses any other record.
- `license.spdx` names the licence; the card gives the details.
- `card` and each artifact `path` are relative to the model directory.
  Artifacts carry the `sha256` and `bytes` of the hydrated file. A file hosted
  on Hugging Face uses `uri` (`hf://<org>/<repo>@<40-hex commit>/<path>`)
  instead of `path`, and must download without credentials, which CI checks.
- `io.streaming` is `stateless`, `explicit_state` (state passed as inputs and
  outputs, listed in `state_pairs`) or `internal_state` (state kept inside the
  model).
- A state pair names the input and output tensors (`{"in": ..., "out": ...}`).
  A stream starts from real-valued zero, which a quantized tensor stores as its
  zero point.
- A golden's `kind` is `single`, `batch` or `sequence`. `batch` and
  `sequence` goldens add a leading axis of `steps`, and a sequence lists in
  `resets` the steps where its state returns to the reset value. CI checks the
  state carry of every sequence exactly and replays every step.
- A golden's `runtime` is the pinned LiteRT it replays with
  (`ai-edge-litert==2.1.2`, as `tools/golden-requirements.txt` pins it);
  `--replay` runs only with that version installed.

Generate each digest from the hydrated file bytes, not from a Git LFS pointer:

```bash
shasum -a 256 path/to/model.tflite path/to/golden.npz
```

On systems with GNU coreutils, `sha256sum` is equivalent.

## 5. Validate the complete corpus

Run validation from a hydrated checkout using the pinned environment:

```bash
git lfs pull
. .golden-venv/bin/activate
python tools/validate_corpus.py corpus-manifest-v1.json
python -m pip install --no-deps -e .
helia-zoo validate --replay
```

Validation rejects unresolved LFS pointers, path escapes, missing artifacts,
digest mismatches, duplicate IDs, missing or unexpected NPZ keys,
signature-incompatible shapes or dtypes, declared tensors that differ from the
model, goldens that do not replay exactly, and v1 entries that disagree with
the records.

If an existing golden changes, summarize representative and maximum numerical
differences in the pull request and obtain approval from the model/corpus
owner.

## 6. Hand off the merged commit

After the model-zoo pull request merges, record the exact commit SHA. In
helia-aot, update `HELIA_MODEL_ZOO_SHA` in
`.github/workflows/release-model-e2e.yml` and add a YAML case referencing the
new stable ID. Do not point helia-aot at an unmerged branch or a moving tag.

The helia-aot procedure is documented in
[Add a model to release testing](https://github.com/AmbiqAI/helia-aot/blob/main/docs/how-to/add-release-model.md).
