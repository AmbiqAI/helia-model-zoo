# sleepKIT five-stage classification — publisher source

Selected artifact SHA256 `c3e37e74c0ffc21b054744d65f2f845f45a9f3257badb15dffe9e6e9371df7d8`, 91664 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

The selected graph is available from the original publisher at [sleepkit/stage/ss-5-tcn-sm/v1.0/model.tflite](https://ambiqai-model-zoo.s3-us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/v1.0/model.tflite). A fresh anonymous download matches the selected SHA256 and size above. Zoo stores its record and typed IO; it does not mirror this model.

Model-weight redistribution terms and a separately identified trained checkpoint remain unresolved. The record's weight license is null. Public publisher access does not grant redistribution permission or benchmark/task qualification. This is a default metadata record whose model is fetched from the publisher with mandatory SHA256 and byte-size checks; no metrics or quality claim is inherited.

The `v1.0` pathname does not establish object immutability. The resolver rejects replacement bytes rather than silently accepting another model. An explicit S3 version request returned AccessDenied; version-ID query URLs are also rejected by the existing resolver. No permission or parser workaround is used. See [source hosting](../../docs/how-to/source-hosting.md) for exact reusable objects and publication gaps.

## Complete diagnostic IO

Install a published Zoo revision containing this record. Use `get('sleepkit-stage').fetch('a8w8', cache=Path('cache'), anonymous=True)` or a pinned `zoo://sleepkit-stage/a8w8@<full published revision>` reference. The HTTPS model source does not need a supplied local graph or workstation root.

The separately supplied companion is the existing 5076-byte diagnostic NPZ, SHA256 `b2469b3030265953eaaf09de051d0e68cf0d138ce099389924fd45820a68519c`. It contains `input_0` INT8 `[1,240,14]` and `output_0` INT8 `[1,240,5]` in graph order. Every input value is the input's quantized real zero, -12. Saved output values are diagnostics, NOT_REFERENCE_GOLDEN; no golden field is supplied. Restore the entire input for each fixed step and capture all outputs. This fixture does not represent task/streaming/numerical qualification or a new model export.

A recipe must bind this exact supplied NPZ as a separate standard Artifact. It is not in the repository or wheel and has no verified public fixture URI. record@1 lists model/card/reference-golden files; this unassessed diagnostic must not be registered as a golden or inserted as an unlisted repository artifact. The model fetch is portable now; saved-fixture portability remains an explicit publication gap. Keep the existing named fixture until an accepted replacement is retrieved and consumers are repointed.

Remaining source limits: separate checkpoint/weight terms, frontend/task policy and long-term immutable publisher placement. Current exact publisher retrieval is available; source availability is not acceptance of the model's predictions.
