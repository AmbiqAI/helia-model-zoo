# HeartKit ECG arrhythmia

Selected trained A8W8 export: 76,592 bytes, SHA256 `75128e080d1ff33cf90dc985a25b5857e939225f0458bc8845d249de252423df`. All external INT8 tensors and their individual quantizers are in `record.json`; the graph is stateless.

## Lineage and training

The trained [HeartKit source](https://github.com/AmbiqAI/heartkit/tree/64cd51b3e1d82c4a325ae2097fe245fc839dea36) checkpoint `results/arr-4-eff-sm/model.keras` has SHA256 `7d570be4e1f4911e38534945dd98fbd5001bb35ec3c72da8233d801df354548e`. This record describes its derived INT8 graph, not the Keras checkpoint. The retained export used helia-edge `fd4dc17081bb1418a972e026fa2874a57fa4fef6`, TensorFlow 2.21.0, Keras 3.15.1 and the strict INT8 converter. Calibration used the first saved 256 validation examples from the trained source; calibration data is not redistributed. No new calibration, conversion or task assessment is represented here.

The checkpoint and selected graph bind weight lineage. Synthetic diagnostic input/output allocations in the benchmark are separate and are not reference goldens. No golden is supplied by this record; no numerical, task-quality, fit, runtime or timing acceptance is inherited. Additional real A16W8, native FP16 and FP32 graphs have their own IO definitions in this record. The separate FP32-compute/FP16-weight variant stays an explicit source intake because the current native-half precision validator cannot represent it.

## Tier, retrieval and rights

Tier: exact selected graph and complete IO metadata, published through this repository's Git LFS route. The record and artifact are available at `zoo://heartkit-arr/a8w8@db261339e5e2a944090b8459a6eb838249bb0d93`; the normal resolver verified anonymous retrieval from an empty cache. A local checkout/root is optional. This source availability does not establish numerical or task qualification.

The trained weights are BSD-3-Clause under the retained publisher-owner declaration, matching the source repository licence. Preserve [the upstream BSD notice](../../licenses/HeartKit-BSD-3-Clause.txt) when redistributing artifacts and retain attribution to Ambiq. This record does not change the original dataset terms.

## Trained parent retrieval

The trained checkpoint is available at [https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.keras), 583939 bytes, SHA256 `7d570be4e1f4911e38534945dd98fbd5001bb35ec3c72da8233d801df354548e`. This publisher `latest` URL is mutable: require the checksum and size, never accept a replacement automatically. Architecture/configuration are separately pinned to HeartKit `64cd51b3e1d82c4a325ae2097fe245fc839dea36`; the checkpoint is not a versioned GitHub `results/` file. Exact byte regeneration also requires the original calibration inputs and converter environment.

## Diagnostic initialization for additional precisions

For each actual graph, initialize all inputs at real-valued zero: floating0 or the individual quantized zero point, with exactly that precision's shape and dtype. Restore inputs before each fixed-step Invoke and capture all outputs. Independent rawzero output allocations are diagnostic, not reference goldens. No A8 fixture is reused across precision. This policy does not qualify streaming or task accuracy.
