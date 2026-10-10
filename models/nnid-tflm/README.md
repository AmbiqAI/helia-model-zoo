# nnid-tflm — source-only intake

Selected artifact SHA256 `06a439de41f80ce1e959a1f0d98277d07bb05adf413d4cdc69c1eb810145927d`, 120480 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

{"basis": "nnid_8x8.tflite distributed by AmbiqAI/nnid_tflm; original checkpoint unknown", "original_checkpoint": null, "status": "publisher_distributed_weights"}

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Complete diagnostic IO

Load the checkout’s `intake-records` parent directory with `load_manifest(checkout / "intake-records")` and select `nnid-tflm`. `Record.fetch("a8w8", cache=<caller cache>, anonymous=True)` retrieves the checksum-pinned original publisher HTTPS artifact. HTTPS resolution does not use a supplied local `root`; bind any separately supplied local copy as a standard direct Artifact instead. Do not claim an unavailable repository mirror.

The companion `fixtures/nnid-tflm/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: nnid-tflm: state pair 0 (serving_default_input_3:0 <- StatefulPartitionedCall:2): dtype or zero point differs

Both explicit feedback pairs are untied. Pair0 differs in zero point and scale; pair1 has input scale513.9821166992188 and output scale0.0075570158660411835. The ordinary record signature/state check reports pair0’s zero-point mismatch; the existing exact-carry/sequence gate rejects both scale mismatches. Preserve both quantizers and refusals; direct cache carry is not admitted, even if pair0 were corrected.
