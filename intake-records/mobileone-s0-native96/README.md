# mobileone-s0-native96 — source-only intake

Selected artifact SHA256 `e923877b35fc26bbbb14b3607e004addbbd949056d3ed44d97880f7493a27ebe`, 2363696 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

Declared trained MobileOne S0 checkpoint SHA256 74080c64f108a15d9c73129f5ae8f8315981c8dc31e111e74b3ae36c7aea2033; complete1000logits at native96 input, distinct from the original224 task condition. Synthetic diagnostic calibration, not task qualification.

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Complete diagnostic IO

Load this directory with existing `load_manifest(path)`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/mobileone-s0-native96/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/mobileone-s0-native96/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.
