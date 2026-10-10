# WeKWS DS-TCN Chinese two-phrase checkpoint — source-only intake

Selected artifact SHA256 `6598c108110074f37dd54d32940ba2a93772222f1dffeb9aae4ed3d5c3ce9e94`, 361168 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

Neural-only DSTCN checkpoint f9ceaede7fbf4735b4b575afca39165841fecc06bcfd29e3e0238915049c4697, original ONNX3e19001f92c3f0373f739ab8859e53a4b2c7bc0230aa455cc479d1609bb62628. Full cache preserved; fbank and source CMVN preparation are external. Old CMVN-containing graph and numerical failures are not qualified by this record.

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Complete diagnostic IO

Load the checkout’s `intake-records` parent directory with existing `load_manifest(checkout / "intake-records")`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/wekws-dstcn/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/wekws-dstcn/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.

The graph input named `cache` is the feature tensor [1,1,40]; the graph input named `input` is the saved cache [1,256,105], paired with `r_cache`. Preserve the recorded order and roles instead of guessing from names.
