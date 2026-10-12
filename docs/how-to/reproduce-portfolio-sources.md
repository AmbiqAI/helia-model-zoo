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

This returns a header, not a model file. The exact selected graph and diagnostic fixture are also available through the caller-authorized private overlay. For literal embedded bytes, the maintained [extraction command](use-supplied-sources.md) verifies the supplied header and extracted model separately. Original publisher readability and a code licence do not establish derivative weight redistribution rights. Local data roots and retained diagnostic NPZ remain caller-supplied inputs, not portable hosting. Ordinary public current-key S3 URLs are usable with expected SHA256 and size. The separate SleepKit source is published at dff63db and merged as56d56b10; the specific denied versionId request remains stopped.

## Full selected-family source dispositions

These are source routes for the existing selected families, not engine/operator/placement admission rules. The original public records and the later authenticated graph/record/diagnostic transfers coexist. A private transfer makes the selected bytes retrievable to an authorized caller; it does not resolve public weight redistribution terms or make its diagnostic output a reference golden. Caller-owned overlay and immutable FileRef bindings remain separate from this public catalog.

The source transfers preserve exact selected identities. The four-variant supplement supplies HeartKit Float32-compute/Float16-storage variants and two distinct Silero alternatives through normal FileRef/Artifact routes; it does not add a Zoo precision or bypass its validators. HeartKit mixed-storage variants omit the incompatible runner record, and the two distinct Silero A8 graphs remain separate Artifacts.

| Selected family | Existing source route | Remaining source/access condition |
|---|---|---|
| mlperf-ad | public mlperf-tiny-ad01 record and pinned graph/golden | Choose the exact selected precision; diagnostic/quality claims stay separate |
| mlperf-kws | public mlperf-tiny-kws record and pinned graph/golden | No private authentication required |
| mlperf-resnet8 | public mlperf-tiny-resnet8 record and pinned graph/golden | Keep the selected wide ResNet graph distinct |
| mlperf-vww | public mlperf-tiny-vww record and pinned graph/golden | Keep the selected graph distinct from other VWW exports |
| heartkit-seg | public graph variants; authenticated exact mixed-storage supplement | Full-IO diagnostic fixture bound separately; parent rights/provenance remain in the card |
| heartkit-arr | public graph variants; authenticated exact mixed-storage supplement | Full-IO diagnostic fixture bound separately; parent rights/provenance remain in the card |
| sleepkit-apnea | authenticated exact selected graph/record/full-IO diagnostic | Selected graph differs from publisher S3 export; no substitution. Unknown public weight terms remain unknown |
| compressionkit-ppg-encoder | public graph record and retained full-IO data | Existing weight-rights/source binding retained |
| microwakeword-okay-nabu | published selected graph and exact retained transform companion | Preserve all six cache pairs, source reset quantizers and notices; re-export is not a retrieval prerequisite |
| speech-to-intent-micro | public pinned source header; authenticated exact selected graph/record/full-IO diagnostic | Header extraction is separate from fetching a raw graph; public derivative weight terms remain unknown |
| gtcrn-official-stream-dns3 | exact supplied selected graph and complete raw-state fixture | No standard remote selected record established; retain named caller inputs. Historical execution hold is not a source availability rule |
| fastenhancer-t | exact supplied selected graph and complete raw-state fixture | No standard remote selected record established; retain named caller inputs. Historical execution hold is not a source availability rule |
| mcunet-in1 | authenticated exact native96 selected graph/record/full-IO diagnostic | Publisher parent is a different graph; conversion/re-export provenance is separate from retrieval |
| circlematch-d32 | removed selection; original source retained separately | No portable selected graph binding; do not restore catalog/execution admission from a source asset |
| rnnoise | public graph/golden record | Exact source identity and source-reset behavior retained; qualification separate |
| wekws-dstcn | authenticated exact neural-only graph/record/full-IO diagnostic; original checkpoint companion | Keep old CMVN graph distinct. Retain all six saved cache inputs; source retrieval does not close historical numerical failures |
| compressionkit-ecg-prior | checksum-pinned original-publisher graph record | Complete 256-token INT32 context and all 256x256 outputs; unknown weight terms, no mirror claim |
| sleepkit-stage | normal catalog, exact checksum-pinned ordinary public S3 graph | Diagnostic NPZ separately bound. Denied versionId route stays stopped; terms/checkpoint regeneration gaps explicit |
| nnid-tflm | public pinned original graph; retained full-state fixed-step diagnostic | Source reset and graph-traced cache pairing are source facts; carry conversion/runtime admission belong to consumers |
| dtln-tflm | public graph record and retained state/initializer source facts | Preserve original variable buffers/reset literals; no fabricated initializers or runtime support claim |
| mobilenetv2-035-96 | authenticated exact A8W8 graph/record/full-IO diagnostic | Original oversized FP32 graph remains a different condition; public weight terms/regeneration provenance separate |
| sanotts-heartnano | shelved; source retained separately | No portable exact selected graph pin established; source availability does not authorize invoking or repairing source UB |
| tinytts | shelved; source retained separately | No standard selected record established; source availability/execution disposition remain distinct |
| silero-v6 | original private record/graph; exact diagnostic and distinct-variant supplement | Original selected, maintained A16 and two distinct A8 identities/fixtures remain separate; no inherited metrics/public visibility |
| firered-vad-stream | original private record/graph plus exact saved diagnostic transfer | Preserve all named cache/index mappings and individual zero points; authorization required |
| mixtoken-relu-imu | authenticated exact synthetic/untrained graph/record/full-IO diagnostic | Source/seed/calibration companions retained; public derivative terms unknown, no trained claim |
| microdpad-derived-fusion | authenticated exact A8/A16 graphs/records/full-IO diagnostics | Each precision has its own graph, quantizers and fixture; source/seed/calibration/unknown terms retained |
| fomo-cuedc-animals | checksum-pinned original anonymous publisher graph | Source-only access does not grant redistribution rights or inherit a saved diagnostic/golden/timing |
| yamnet256-fsd5 | authenticated exact selected graph/record/full-IO diagnostic | Trained parent/FSD5 head/frontend/re-export provenance and public weight terms remain separate gaps |
| acdnet20-esc50 | authenticated exact selected graph/record/full-IO diagnostic | ESC50 parent/frontend/re-export provenance and public weight terms remain separate gaps |
| e2a2-clean-speech-op4 | repository-only intake with named selected graph | Selected INT8 output lacks a quantizer/full saved fixture; retain actual preparation gap, do not invent a scale |
| efficientat-mn02 | authenticated exact normalized selected graph/record/full-IO diagnostic | Preserve full10s input/both outputs; original/normalized graphs distinct. Checkpoint/re-export provenance separate |
| mobileone-s0-native96 | authenticated exact native96 selected graph/record/full-IO diagnostic | Original224 graph/quality separate; preserve selected hash/full1000 outputs and unknown public weight terms |
| cp-mobile-dcase2025 | authenticated exact general-submodel graph/record/full-IO diagnostic | Preserve complete frontend input/ten logits; checkpoint/re-export provenance separate from obtainable bytes |

[Supplied sources](use-supplied-sources.md) describes exact SoundKit/SE2 files, embedded-header extraction and station/cache prerequisites. Those additional source packets do not silently enlarge this selected-family table or establish runtime qualification.
