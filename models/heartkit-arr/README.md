# HeartKit ECG arrhythmia

Selected trained A8W8 export: 76,592 bytes, SHA256 `75128e080d1ff33cf90dc985a25b5857e939225f0458bc8845d249de252423df`. All external INT8 tensors and their individual quantizers are in `record.json`; the graph is stateless.

## Lineage and training

The trained [HeartKit source](https://github.com/AmbiqAI/heartkit/tree/64cd51b3e1d82c4a325ae2097fe245fc839dea36) checkpoint `results/arr-4-eff-sm/model.keras` has SHA256 `7d570be4e1f4911e38534945dd98fbd5001bb35ec3c72da8233d801df354548e`. This record describes its derived INT8 graph, not the Keras checkpoint. The retained export used helia-edge `fd4dc17081bb1418a972e026fa2874a57fa4fef6`, TensorFlow 2.21.0, Keras 3.15.1 and the strict INT8 converter. Calibration used the first saved 256 validation examples from the trained source; calibration data is not redistributed. No new calibration, conversion or task assessment is represented here.

The checkpoint and selected graph bind weight lineage. Synthetic diagnostic input/output allocations in the benchmark are separate and are not reference goldens. No golden is supplied by this record; no numerical, task-quality, fit, runtime or timing acceptance is inherited. Other precision variants remain separate and are not supplied here.

## Tier, retrieval and rights

Tier: exact selected graph and complete IO metadata, prepared for the existing repository Git LFS route. The local graph is staged alongside its record. Hosted retrieval is pending publication and authorized artifact transfer; this is not an available public revision reference. Use the caller's explicit checkout/root for local preparation.

The trained weights are BSD-3-Clause under the retained publisher-owner declaration, matching the source repository licence. Preserve [the upstream BSD notice](../../licenses/HeartKit-BSD-3-Clause.txt) when redistributing artifacts and retain attribution to Ambiq. This record does not change the original dataset terms.
