# Reproduce selected portfolio sources

The selected portfolio and the installed catalog are different sets. Keep every selected family and its actual precision/graph identity visible in the consumer's existing selection matrix, including held, shelved and removed entries. Public model names are not graph identity: select by the model SHA256 and exact IO/fixture. Normal recipes may use a standard Zoo reference, an original-publisher Artifact, or named supplied local data. A pending source is not an unsupported model.

## Normal installed catalog

Install the package from the exact published repository revision used by the recipe. `records()`, `get()` and `resolve("zoo://<id>/<precision>@<full revision>")` use the maintained API and normal cache. Existing MLPerf4 A8W8 graph/golden pairs, RNNoise, HeartKit A8 and compressionKIT PPG are available at db261339e5e2a944090b8459a6eb838249bb0d93. Newer precision/source entries require their own published revision. Hosted graph retrieval does not establish numerical, task, streaming or device acceptance.

The HeartKit records expose A8W8, actual A16W8, native Float16 and Float32 graphs with separate typed IO. Float32 compute with Float16 stored weights is a distinct source intake because the existing native-half precision validator refuses it; it is not substituted for native Float16. Weight rights and exact trained parents are documented in the cards. Publisher mutable `latest` checkpoint paths must be constrained by SHA256 and byte size.

FOMO and the ECG prior use their original checksum-pinned publisher artifact without a graph mirror; unresolved weight terms remain null. The ECG prior preserves the complete 256-token INT32 context and all 256×256 INT8 logits, with mixed INT8/Float32 internals. microWakeWord preserves all six explicit cache pairs and signed score mapping; its pinned model-weight licence and modification notices are supplied. Frontend/task policies remain separate from the model's full typed IO.

## Named supplied sources and diagnostic IO

[Temporary intake records](../../intake-records/README.md) use the same record@1 parser and explicit `load_manifest(checkout / "intake-records")`. They are repository-only metadata and excluded from the default installed catalog. The consumer must explicitly bind each record or named Artifact and its remaining availability condition; do not silently ignore the directory or count it as fully migrated hosting.

A supplied root contains `models/<id>/<precision>/model.tflite` and the caller's checksum-pinned full-IO NPZ. Use standard Artifact path/hash/size definitions in the existing recipe workflow; no new source registry or downloader is needed. Restore every saved input, including all cache/state inputs, before each fixed-step diagnostic Invoke, and capture every output. For new zero-input cases, initialize integer signals at their individual quantized zero point, floating signals at0, and categorical token IDs at the explicitly stated token ID. Each different precision uses its own dtype/shape/initialization; do not reuse an A8 fixture for A16 or floating IO.

Diagnostic output allocations are NOT_REFERENCE_GOLDEN. A saved fixture or successful source fetch is not a streaming/quality claim. Original public goldens retain their own checksum and semantics; synthetic diagnostic NPZ must not replace them. Private selected derivatives remain private even when their upstream architecture is public; use the authorized normal overlay or named internal Artifact and retain exact selected identity.

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
| `sleepkit-apnea` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `compressionkit-ppg-encoder` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `microwakeword-okay-nabu` | default source record; artifact at repository LFS or original publisher | New reviewed-head/LFS publication pending; exact transform source retained; portable generator packaging pending; frontend/task/streaming qualification remains separate |
| `speech-to-intent-micro` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `gtcrn-official-stream-dns3` | held | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `fastenhancer-t` | held | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `mcunet-in1` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `circlematch-d32` | removed | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; No portable selected graph pin; retained original source asset is separate |
| `rnnoise` | default source record; artifact at repository LFS or original publisher | Exact source and diagnostic IO; qualification separate |
| `wekws-dstcn` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `compressionkit-ecg-prior` | default source record; artifact at repository LFS or original publisher | weight terms unknown; no mirror/public benchmark admission; source metadata/typed diagnostic initialization supplied |
| `sleepkit-stage` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `nnid-tflm` | temporary repository-only intake; named local Artifact or original publisher | Both feedback pairs have untied quantizers; pair0 zero-point mismatch and both exact-carry scale refusals retained |
| `dtln-tflm` | default source record; artifact at repository LFS or original publisher | Weight redistribution terms unknown; four internal mutable-state initialization/fixture still missing |
| `mobilenetv2-035-96` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `sanotts-heartnano` | shelved | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; No portable selected graph pin; retained original source asset is separate |
| `tinytts` | shelved | No normal pinned zoo record for selected graph; Selected artifact redistribution/license binding not recorded here; Current BENCH execution catalog has no exact selected graph entry |
| `silero-v6` | private record and named internal data; no public metadata/URI | Private managed-source visibility unchanged; exact model/fixture copied to named internal data, no public mirror |
| `firered-vad-stream` | private record and named internal data; no public metadata/URI | Private managed-source visibility unchanged; exact model/fixture copied to named internal data, no public mirror |
| `mixtoken-relu-imu` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `microdpad-derived-fusion` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `fomo-cuedc-animals` | default source record; artifact at repository LFS or original publisher | weight terms unknown; no mirror/public benchmark admission; source metadata/typed diagnostic initialization supplied |
| `yamnet256-fsd5` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `acdnet20-esc50` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `e2a2-clean-speech-op4` | temporary repository-only intake; named local Artifact or original publisher | Selected INT8 output has no quantizer; no full saved fixture. Existing preparation refusal retained, no scale invented. |
| `efficientat-mn02` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `mobileone-s0-native96` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
| `cp-mobile-dcase2025` | temporary repository-only intake; named local Artifact or original publisher | Exact selected graph redistribution/managed hosting unresolved; default public catalog admission pending |
