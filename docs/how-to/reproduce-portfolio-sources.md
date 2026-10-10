# Reproduce selected portfolio sources

The selected portfolio and the installed catalog are different sets. Keep every selected family and its actual precision/graph identity visible in the consumer's existing selection matrix, including held, shelved and removed entries. Public model names are not graph identity: select by the model SHA256 and exact IO/fixture. Normal recipes may use a standard Zoo reference, an original-publisher Artifact, or named supplied local data. A pending source is not an unsupported model.

## Normal installed catalog

Install the package from the exact published repository revision used by the recipe. `records()`, `get()` and `resolve("zoo://<id>/<precision>@<full revision>")` use the maintained API and normal cache. Existing MLPerf4 A8W8 graph/golden pairs, RNNoise, HeartKit A8 and compressionKIT PPG are available at db261339e5e2a944090b8459a6eb838249bb0d93. Newer precision/source entries require their own published revision. Hosted graph retrieval does not establish numerical, task, streaming or device acceptance.

The HeartKit records expose A8W8, actual A16W8, native Float16 and Float32 graphs with separate typed IO. Float32 compute with Float16 stored weights is a distinct source intake because the existing native-half precision validator refuses it; it is not substituted for native Float16. Weight rights and exact trained parents are documented in the cards. Publisher mutable `latest` checkpoint paths must be constrained by SHA256 and byte size.

FOMO and the ECG prior use their original checksum-pinned publisher artifact without a graph mirror; unresolved weight terms remain null. The ECG prior preserves the complete 256-token INT32 context and all 256×256 INT8 logits, with mixed INT8/Float32 internals. microWakeWord preserves all six explicit cache pairs and signed score mapping; its pinned model-weight licence and modification notices are supplied. Frontend/task policies remain separate from the model's full typed IO.

SleepKit stage has a normal-catalog binding to the exact original publisher graph. The ordinary public S3 URL is accepted only with the recorded SHA256 and byte size; a fresh normal fetch verified those selected bytes. The specific denied versionId request remains stopped. Its [card](../../models/sleepkit-stage/README.md) binds a separately supplied diagnostic NPZ that remains outside the catalog's reference goldens. [Source hosting](source-hosting.md) lists the reusable HeartKit parents, different publisher graph exports and exact missing publication conditions. Historical public readability does not change unknown weight terms.

## Named supplied sources and diagnostic IO

[Temporary intake records](../../intake-records/README.md) use the same record@1 parser and explicit `load_manifest(checkout / "intake-records")`. They are repository-only metadata and excluded from the default installed catalog. The consumer must explicitly bind each record or named Artifact and its remaining availability condition; do not silently ignore the directory or count it as fully migrated hosting.

A supplied root contains `models/<id>/<precision>/model.tflite` and the caller's checksum-pinned full-IO NPZ. Use standard Artifact path/hash/size definitions in the existing recipe workflow; no new source registry or downloader is needed. Restore every saved input, including all cache/state inputs, before each fixed-step diagnostic Invoke, and capture every output. For new zero-input cases, initialize integer signals at their individual quantized zero point, floating signals at0, and categorical token IDs at the explicitly stated token ID. Each different precision uses its own dtype/shape/initialization; do not reuse an A8 fixture for A16 or floating IO.

Diagnostic output allocations are NOT_REFERENCE_GOLDEN. A saved fixture or successful source fetch is not a streaming/quality claim. Original public goldens retain their own checksum and semantics; synthetic diagnostic NPZ must not replace them. Private selected derivatives remain private even when their upstream architecture is public; use the authorized normal overlay or named internal Artifact and retain exact selected identity.

## Original source companions

The source distribution includes `intake-records` and this guide. The wheel's default catalog remains separate: inspect intake metadata explicitly with `load_manifest(source_root / "intake-records")`. Five intake records use the existing optional `upstream` object to bind an original source file or checkpoint. Its hash identifies that file, not the selected TFLite derivative; full graph/IO pins stay in `precisions`. Source cards state file type, size, transport and reproduction gaps. No new schema, downloader or implicit catalog admission is needed.

For the Speech-to-Intent companion, the normal client retrieves the C header by its own hash and size:

```python
from helia_model_zoo.hydrate import fetch_file
from helia_model_zoo.manifest import FileRef

header = fetch_file(FileRef(
    "https://raw.githubusercontent.com/AIWintermuteAI/Speech-to-Intent-Micro/"
    "7e6674117c55145d0071d39768c183e63e917479/"
    "inference_code/Wio_Terminal/wio_speech_to_intent/slu_model.h",
    "72a7d1a11e51879eff759c7cbdd0e00b2f654a374b94658a44dfa692eaa33272",
    247602,
), anonymous=True)
```

This returns a header, not a model file. Use the already supplied graph for model execution until an authorized raw-graph transfer is available. Original publisher readability and a code licence do not establish derivative weight redistribution rights. Local data roots and retained diagnostic NPZ remain caller-supplied inputs, not portable hosting. Ordinary public current-key S3 URLs are usable with expected SHA256 and size. The separate SleepKit source is published at dff63db and merged as56d56b10; the specific denied versionId request remains stopped.

## Full selected-family dispositions

This table documents the existing selected families; it is not another authoritative catalog or a claim that all artifacts are public. Precision/graph selection remains in the existing consumer catalog and matrix. Sources with incomplete terms or preparation retain named local routes and explicit gaps.

| Selected family | Source route | Remaining condition |
| --- | --- | --- |
| `mlperf-ad` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `mlperf-kws` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `mlperf-resnet8` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `mlperf-vww` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `heartkit-seg` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `heartkit-arr` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `sleepkit-apnea` | local selected graph; repository-only intake | Selected graph differs from publisher S3 export; do not substitute it. Exact selected graph hosting/terms remain unresolved. Checkpoint/config/calibration bindings describe regeneration separately and do not block an obtainable exact graph |
| `compressionkit-ppg-encoder` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `microwakeword-okay-nabu` | published selected graph at `68f3b256`; pinned public parent | Exact transform source retained separately; portable generator packaging and byte-regeneration check pending. Frontend/task/streaming qualification separate |
| `speech-to-intent-micro` | public pinned source header; local selected graph | Embedded bytes match selected graph; normal model resolver cannot decode C header. Raw graph transfer and weight terms pending |
| `gtcrn-official-stream-dns3` | held | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `fastenhancer-t` | held | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `mcunet-in1` | local selected graph; repository-only intake | Publisher MCUNet TFLite is860752B, selected native96 derivative860888B; do not substitute parent. Exact derivative transfer, transform recipe and weight terms missing |
| `circlematch-d32` | removed | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; No portable selected graph pin; retained original source asset is separate |
| `rnnoise` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `wekws-dstcn` | pinned original checkpoint; local selected graph | Checkpoint query API is outside normal HTTPS FileRef grammar. Exact neural-only transform/calibration/environment and graph transfer/weight terms missing; old CMVN route and numerical failures held |
| `compressionkit-ecg-prior` | default source record; artifact at repository LFS or original publisher | weight terms unknown; no mirror/public benchmark admission; source metadata/typed diagnostic initialization supplied |
| `sleepkit-stage` | normal catalog source; checksum-pinned original publisher graph | Exact selected graph is obtainable from the ordinary public S3 URL with expected SHA256/size, published at dff63db and merged as56d56b10. Diagnostic NPZ remains separate from goldens; weight terms and regeneration inputs remain unresolved |
| `nnid-tflm` | pinned original non-S3 graph in repository-only intake | Graph source is independently retrievable; both feedback pairs have untied quantizers. Pair0 zero-point mismatch and both exact-carry scale refusals prevent normal runtime acceptance |
| `dtln-tflm` | default source record; artifact at repository LFS or original publisher | Weight redistribution terms unknown; four internal mutable-state initialization/fixture still missing |
| `mobilenetv2-035-96` | local selected A8W8 identity with INT8 external IO; repository-only intake | Exact selected graph hosting and weight terms pending. Original oversized FP32 evidence remains separate; checkpoint/freeze/export/calibration bindings describe regeneration |
| `sanotts-heartnano` | shelved | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; No portable selected graph pin; retained original source asset is separate |
| `tinytts` | shelved | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `silero-v6` | private record and named internal data; no public metadata/URI | Private managed-source visibility unchanged; exact model/fixture copied to named internal data, no public mirror |
| `firered-vad-stream` | private record and named internal data; no public metadata/URI | Private managed-source visibility unchanged; exact model/fixture copied to named internal data, no public mirror |
| `mixtoken-relu-imu` | local synthetic/untrained graph; repository-only intake | Random-weight seed/config/source and exact calibration recipe need a portable binding; establish derivative rights and raw graph transfer; no trained claim |
| `microdpad-derived-fusion` | local synthetic/untrained graph; A8/A16 repository-only intake | A16 companion metadata uses its distinct graph/IO pins; random-weight source/config/calibration and derivative rights/transfer unresolved |
| `fomo-cuedc-animals` | default source record; artifact at repository LFS or original publisher | weight terms unknown; no mirror/public benchmark admission; source metadata/typed diagnostic initialization supplied |
| `yamnet256-fsd5` | local selected graph; repository-only intake | Exact trained parent/checkpoint, FSD5 head/frontend and conversion/calibration recipe unbound; weight terms and raw graph transfer missing |
| `acdnet20-esc50` | local selected graph; repository-only intake | Exact ESC50 checkpoint, full frontend/source/export/calibration recipe unbound; weight terms and raw graph transfer missing |
| `e2a2-clean-speech-op4` | temporary repository-only intake; named local Artifact or original publisher | Selected INT8 output has no quantizer; no full saved fixture. Existing preparation refusal retained, no scale invented. |
| `efficientat-mn02` | pinned public source README; retained checkpoint and local graph | Bind checkpoint file/size and full10s source adaptations/calibration/converter recipe; graph transfer/weight terms pending. Singleton normalization stays a separate identity |
| `mobileone-s0-native96` | pinned public source; retained fused checkpoint and local graph | Bind native96 adaptation/calibration/converter recipe; raw graph transfer and weight terms pending. Original224 quality is separate |
| `cp-mobile-dcase2025` | pinned official checkpoint; local selected graph | General submodel, complete frontend/ten-logit output view, calibration and converter recipe required; raw graph transfer and weight terms pending |
