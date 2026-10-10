# fomo-cuedc-animals — publisher-hosted source

Selected artifact SHA256 `23449b2890a4afedbcfbcc4ad11810f70c6dd5fdb3da23beb0aad0a7a47e5263`, 56872 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

{"basis": "Retained MODEL-SOURCE.json and exact published trained.tflite", "status": "publisher_declared_trained"}

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; the original trained graph stays at its pinned publisher, without a repository mirror. This source-only record reads the original checksum-pinned publisher artifact. Its metadata can be discovered by default records(); unknown weight terms still block mirroring and public benchmark admission. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Complete diagnostic IO

Load this directory with existing `load_manifest(path)`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/fomo-cuedc-animals/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/fomo-cuedc-animals/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.

Publisher source: CUEDC animals FOMO trained FlatBuffer at43c55f1c. Selected graph is the same original23449b2890a4, not a newly converted derivative. The full16×16×5output heatmap is retained. Source weights are declared trained by the publisher; separate original checkpoint and weight redistribution terms are not bound. Resolve this source record at its exact repository revision and retain the publisher artifact checksum.
