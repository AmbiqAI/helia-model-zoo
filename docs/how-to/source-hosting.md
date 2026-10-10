# Publisher sources and remaining payload gaps

Keep Zoo records, model cards, typed IO, source/rights notices and checksums in this repository. Payloads may be fetched from an existing original publisher using the normal record@1 resolver. Match the SHA256 and byte size; a model name or ETag is not identity evidence. Source availability does not establish numerical, task, streaming or device qualification.

## Existing exact matches

Anonymous ordinary GETs retrieved these exact bytes. The S3 `v1.0` key is a version-named path, not proof of enforced immutability. Use the mandatory checksum and size; replacement bytes must fail closed.

| Source | Existing public object | SHA256 | Bytes | Use |
| --- | --- | --- | --- | --- |
| SleepKit stage | [graph](https://ambiqai-model-zoo.s3-us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/v1.0/model.tflite) | `c3e37e74c0ffc21b054744d65f2f845f45a9f3257badb15dffe9e6e9371df7d8` | 91664 | Default `sleepkit-stage/a8w8` source record; null weight terms, no mirror |
| HeartKit segmentation | [trained parent](https://ambiqai-model-zoo.s3-us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/v1.0/model.keras) | `80481fc0140e92f6783f9f59f41c4e88563c802299f98e6d22480647a27fb031` | 284587 | Exact source checkpoint; distinct from selected LiteRT graphs |
| HeartKit rhythm | [trained parent](https://ambiqai-model-zoo.s3-us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/v1.0/model.keras) | `7d570be4e1f4911e38534945dd98fbd5001bb35ec3c72da8233d801df354548e` | 583939 | Exact source checkpoint; distinct from selected LiteRT graphs |

The two HeartKit parent identities retain their existing owner weight notice and source/calibration requirements in the [segmentation](../../models/heartkit-seg/README.md) and [rhythm](../../models/heartkit-arr/README.md) cards. SleepKit model-weight terms and a separate trained checkpoint are unresolved; public readability is not a redistribution grant. No model weights were added or removed here.

## Normal SleepKit access

After installing a published Zoo revision containing the entry, the model needs no workstation data root:

```python
from pathlib import Path
from helia_model_zoo import get

record = get("sleepkit-stage")
model = record.fetch("a8w8", cache=Path("cache"), anonymous=True)
```

Use a full published repository revision for reproducible recipe metadata, for example `zoo://sleepkit-stage/a8w8@<full published revision>`. A local unpushed branch is not a remotely available reference.

The separately supplied diagnostic NPZ is 5076 bytes, SHA256 `b2469b3030265953eaaf09de051d0e68cf0d138ce099389924fd45820a68519c`. It holds the complete INT8 input `[1,240,14]` at its quantized real zero -12 and saved INT8 output `[1,240,5]`. The output values are diagnostics, **NOT_REFERENCE_GOLDEN**. Bind the supplied file through the consumer's existing standard Artifact path/hash/size interface. It is not distributed in the repository or wheel and has no verified public fixture URI; do not try to fetch it from a Zoo revision. The graph is portable, while this saved fixture remains a separate source dependency.

record@1 exposes model/card/reference-golden files, and the existing completeness check rejects unlisted tracked NPZ artifacts. Neither a new diagnostic schema field nor a fake golden is introduced. Preserve every input and output and retain the existing named file until an approved normal publisher replacement is retrieved and consumers are repointed. Source checks do not assess predictions or establish carried-stream semantics.

## Smallest missing publication set

Existing publisher graph exports are not substituted for these selected bytes:

| Family | Selected A8W8 SHA256 / bytes | Existing corresponding S3 graph SHA256 / bytes |
| --- | --- | --- |
| HeartKit segmentation | `1d32489a846bad80a9473a76ab403331568cf757cd629813939dab6bd734ae0b` / 30840 | `2c233e397ab6272640b589c91748a5c66f4d884d6828f1c028f130ccfb6ebac6` / 32376 |
| HeartKit rhythm | `75128e080d1ff33cf90dc985a25b5857e939225f0458bc8845d249de252423df` / 76592 | `dc033a856215070d280f550101012230cfda38f0befb307ddf7d998ae8ff3dca` / 69368 |
| SleepKit apnea | `779e1b0dbedfdd4f124c6cb5bb136c287c50bdefd11339eb655c804cc88e5eb1` / 41056 | `8f0b728f4b90413840cd33d0f2485eedac8c93a2cbc38340ced4c371e601c1a4` / 46496 |

Their existing `latest` and `v1.0` publisher exports match one another, but differ from the selected graphs. Keep the selected graph references unchanged. The same publisher prefixes do not separately supply the selected HeartKit A16W8, native FP16, FP32 or FP32-compute/FP16-storage variants:

| Family | Variant | Selected SHA256 | Bytes |
| --- | --- | --- | --- |
| HeartKit segmentation | A16W8 | `91cabb0a57ea1b66c7de78d03e52e94eaa89c4149eb3ed0afa88f0a597d2c626` | 35488 |
| HeartKit segmentation | native FP16 | `8104511d2fc2587aa31aea938139c92ce672c24ba840bcae4ebe5b8078a08685` | 34000 |
| HeartKit segmentation | FP32 | `e72790735c1bf673a485f7ae79b0a45ab614f4ebb32525eb3d37e72cee8e73f0` | 40964 |
| HeartKit segmentation | FP32 compute/FP16 storage | `642cfe5371f89d5bfffb0120f76e492faab66c238e16fe3f88be0ec35105d042` | 34836 |
| HeartKit rhythm | A16W8 | `dae7f312e78c8a4f65b344258d3f92ebf8a7618fa819fbd408296752fe5729c8` | 87832 |
| HeartKit rhythm | native FP16 | `521e63b5ca20a86d2f4077f0c06c223c9c594fca01f6bb50421859908aceac47` | 90312 |
| HeartKit rhythm | FP32 | `9f6c276dd547c4fe649c7cb389ba45773c2ae31b439ec65695937ce5d2cb2a5c` | 129972 |
| HeartKit rhythm | FP32 compute/FP16 storage | `2664008a180d84a031c75a7e6a439eb67610fd7d86cf1c068418b3128be9ca67` | 91416 |

ADK/publisher owners can publish these eleven distinct graph identities at commit-pinned HF paths or immutable/content-addressed S3 keys, retaining the applicable weight notices and source/calibration provenance. The two FP32-compute/FP16-storage artifacts still have their existing precision-policy refusals; hosting does not make them native FP16 or eligible for execution.

Publish checksum-bound complete diagnostic fixtures and explicit checkpoint/source/rights bindings where absent. SleepKit stage still needs its exact 5076-byte diagnostic NPZ at an approved managed source; all existing unique fixture inputs remain in their named data stores. A source-only metadata record does not override missing initialization, quantizers, private visibility, numerical or task gates elsewhere in the selected portfolio.

## Access limits

The public HeartKit and SleepKit current-key listings contain model, Keras checkpoint, configuration, history and metrics objects. They do not list separate record@1 files, typed NPZ companions or model-weight licence/checksum manifests. Two bounded public HF searches under publisher Ambiq for HeartKit/SleepKit returned no repositories; private repositories and other publisher names were not audited.

The first explicit S3 version-ID GET returned `HTTP403 AccessDenied: Access Denied`; that route was stopped without another version request or alternate account/host. The existing HTTPS resolver also rejects query-bearing URLs. This increment only uses the separately authorized ordinary public GET source that already matched the selected graph; it makes no version-ID or immutability claim and changes neither permissions nor the resolver. Keep existing repository/local payloads until any future approved publisher migration has passed normal fresh-cache retrieval and consumer repointing. No upload, history rewrite or payload removal is part of this change.
