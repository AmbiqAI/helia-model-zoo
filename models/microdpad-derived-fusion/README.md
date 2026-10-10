# microdpad-derived-fusion — source-only intake

Selected artifact SHA256 `82b5d4753856dca1306af4a38a7222c84f5eacdc855891a65be595e5c7015ffd`, 261744 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

{"basis": "random", "status": "synthetic_untrained"}

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Complete diagnostic IO

Load the checkout’s `intake-records` parent directory with existing `load_manifest(checkout / "intake-records")`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/microdpad-derived-fusion/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/microdpad-derived-fusion/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.

The existing A16W8 variant retains both full INT16 inputs and the full INT16 output, with INT8 weights and separate model/fixture hashes. It is the same synthetic, untrained family; no trained/task metrics are inherited.
