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
```

Validation rejects unresolved LFS pointers, path escapes, missing artifacts,
digest mismatches, duplicate IDs, missing or unexpected NPZ keys, and
signature-incompatible shapes or dtypes.

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
