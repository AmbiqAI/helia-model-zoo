# Add a model to the heliaAOT release corpus

The model zoo owns artifact identity and integrity. It does not decide which
models helia-aot runs as release coverage. A model becomes eligible for release
testing only after its model, golden fixture, model card, and manifest entry
merge here; a separate helia-aot change then pins that merged commit and adds a
release case.

## 1. Create a self-contained model directory

Place the files under the appropriate domain and family:

```text
<domain>/<family?>/<model>/
  README.md
  model.tflite
  golden.npz
```

The repository's `.gitattributes` tracks `.tflite` and `.npz` files with Git
LFS. Confirm that Git sees new artifacts as LFS objects before committing:

```bash
git check-attr filter -- path/to/model.tflite path/to/golden.npz
git lfs status
```

Add the model to the appropriate domain README and to the root README inventory
so people can discover it independently of the machine-readable manifest.

## 2. Write the model card

The model README is the provenance and license reference used by the manifest.
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
python tools/generate_golden.py path/to/model.tflite path/to/golden.npz --seed 42
```

The NPZ must contain consecutive `input_N` and `output_N` arrays whose shapes
and dtypes match the TFLite signature. If the standard generator is unsuitable,
document the deterministic input-generation method and runtime version in the
pull request and model card.

## 4. Add the manifest entry

Choose a stable, descriptive ID. IDs are an API consumed by helia-aot and
should not be renamed when files move. Add one entry to
`corpus-manifest-v1.json`:

```json
{
  "id": "example-int8",
  "model": "vision/example/model.tflite",
  "model_sha256": "<hydrated model SHA-256>",
  "golden": "vision/example/golden.npz",
  "golden_sha256": "<hydrated golden SHA-256>",
  "reference_runtime": "ai-edge-litert",
  "reference_runtime_version": "2.1.2",
  "provenance_reference": "vision/example/README.md",
  "license_reference": "vision/example/README.md"
}
```

Add the same model to `src/helia_model_zoo/manifest.json` (manifest v2), with
the v1 ID as an alias of its precision:

```json
"aliases": {"example-int8": "int8"}
```

Copy an existing v2 entry as a template. Take each tensor's name, shape, dtype,
scale and zero point from the model, record the LiteRT resolver its golden
replays under (`builtin_ref` for new goldens), and list explicit state pairs
under `io.state_pairs`. CI refuses a v2 entry that disagrees with its v1 alias.

The v2 fields:

- `visibility` must be `public`: this repository is public, and CI refuses any
  other entry.
- `tier` is `converted` for a model we run but do not train (every current
  entry), or `native` for one whose architecture helia-edge can build and train.
- Artifacts use `lfs://` paths with `sha256` and `bytes` of the hydrated file;
  cards and license references use `repo://` paths. An `https://` or `hf://`
  source in this manifest must download without credentials, which CI checks.
- `io.streaming` is `stateless`, `explicit_state` (state passed as inputs and
  outputs, listed in `state_pairs`) or `internal_state` (state kept inside the
  model).
- A state pair names the input and output indices, its `reset` value (`zeros`
  is real-valued zero, which a quantized tensor stores as its zero point), and
  `scales_tied`: whether the two tensors share a scale, so the output can be fed
  back unchanged.
- A golden's `kind` is `single`; `batch` and `sequence` goldens add a leading
  axis of `steps`, and a sequence lists in `resets` the steps where its state
  returns to the reset value. Neither is used yet.
- `--replay` runs only with the golden's `reference_runtime_version` of LiteRT
  installed, as `tools/golden-requirements.txt` pins it.

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
model, goldens that do not replay exactly, and v1/v2 disagreements.

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
