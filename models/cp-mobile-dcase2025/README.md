# cp-mobile-dcase2025 — source-only intake

Selected artifact SHA256 `2b5d12cd34d8df9b33cc4a46d8d5a218c721c6141092ece0830cf6fb612cbaea`, 115480 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

Declared trained general/unknown-device base submodel, source revision da99d532999c8148cf3e0c7e0f9782324c04071e, checkpoint399908d314b99e86b95c96edeea69de42710f23a7a0a53b2161931ed5f2425c8. Full ten scene logits, one-second input; final singleton output Squeeze chain replaced by fixed[1,10]Reshape. No neural weights were replaced.

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Pinned original source companion

The `upstream` field identifies the official trained checkpoint, not this selected TFLite export: `CPJKU/dcase2025_task1_inference@da99d532999c8148cf3e0c7e0f9782324c04071e`, `Schmid_CPJKU_task1/ckpts/baseline.ckpt`, SHA256 `399908d314b99e86b95c96edeea69de42710f23a7a0a53b2161931ed5f2425c8`,5392407bytes. The original source and checkpoint are public upstream inputs; selected graph redistribution terms remain unresolved. Reuse the retained checkpoint with this exact pin. Source reproduction additionally needs the general submodel selection, complete one-second feature frontend, fixed ten-logit output view, converter environment and exact representative calibration inputs. A parent pin alone does not reproduce the selected quantized bytes. No export or fresh checkpoint retrieval is claimed here.

## Complete diagnostic IO

Load the checkout’s `intake-records` parent directory with existing `load_manifest(checkout / "intake-records")`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/cp-mobile-dcase2025/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/cp-mobile-dcase2025/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.
