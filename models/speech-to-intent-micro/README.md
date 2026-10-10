# Speech-to-Intent-Micro Wio default100x13 — source-only intake

Selected artifact SHA256 `d77011e4b68027b41cd1fd3f87d5d28eb3c4a516b2de4050da74a941fb59bb7c`, 40128 bytes. Record@1 contains the full ordered external IO and each tensor's quantizer. `a8w8` identifies this selected variant; inspect typed IO and the limits below rather than assuming every internal graph section is INT8.

## Source, training and rights

{"basis": "No separate trained-checkpoint identity established in source binding", "status": "unknown"}

Unknown model-weight redistribution terms: no public graph mirror or default public catalog admission. Metadata is readable; graph bytes remain a named supplied local source pending hosting/rights, or at their pinned original publisher. This temporary intake is not part of default packaged `records()` and is not a retrievable `zoo://` catalog entry. Unknown weight terms do not imply the publisher's upstream architecture is private. No metrics or quality claim is inherited.

## Pinned original source companion

The `upstream` field binds the original publisher's C header, not a raw TFLite URL: revision `7e6674117c55145d0071d39768c183e63e917479`, path `inference_code/Wio_Terminal/wio_speech_to_intent/slu_model.h`, SHA256 `72a7d1a11e51879eff759c7cbdd0e00b2f654a374b94658a44dfa692eaa33272`,247602bytes. Anonymous normal-client retrieval and static decoding of its byte initializer match the selected40128byte graph exactly; no conversion or model execution is needed for that identity check. The header can be retrieved with existing `FileRef`/`fetch_file` using [the pinned raw URL](https://raw.githubusercontent.com/AIWintermuteAI/Speech-to-Intent-Micro/7e6674117c55145d0071d39768c183e63e917479/inference_code/Wio_Terminal/wio_speech_to_intent/slu_model.h). The normal model resolver does not decode C headers, so selected graph retrieval remains a supplied local Artifact. The remaining direct-graph transfer needs explicit model-weight redistribution terms and a checksum-bound raw TFLite location; no header URL is substituted for the model URL.

## Complete diagnostic IO

Load the checkout’s `intake-records` parent directory with existing `load_manifest(checkout / "intake-records")`, select this record, then call `Record.fetch('a8w8', root=<explicit supplied data root>)`. A relative model path is resolved under `models/speech-to-intent-micro/a8w8/model.tflite` in that caller root. Do not treat a missing local graph as a published repository artifact.

The companion `fixtures/speech-to-intent-micro/io.npz` contains `input_i` and `output_i` in graph order. Restore every supplied input, including all saved cache/state, before each fixed-step Invoke and capture all outputs. Outputs are diagnostic allocations, not reference goldens. No golden field is supplied. The selected NPZ is separately checksum-bound by the caller's standard Artifact descriptor. No numerical/task/streaming qualification or new export is claimed.

Existing source limits: No default catalog/managed exact artifact hosting; separate weight terms remain unresolved.
