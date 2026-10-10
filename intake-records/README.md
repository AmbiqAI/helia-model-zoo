# Temporary source-only records

These repository-only record@1 files describe exact selected artifacts that the default catalog cannot currently supply or admit. They are excluded from the installed package's default `records()`; they are not fully migrated or retrievable Zoo catalog entries. Metadata visibility, original-publisher visibility, weight redistribution permission, support and benchmark eligibility are separate facts. No graph or fixture is mirrored here. Selected private VAD metadata and URIs are held outside this repository.

Read the checkout explicitly using the existing API:

```python
from pathlib import Path
from helia_model_zoo import load_manifest

records = load_manifest(Path("helia-model-zoo") / "intake-records")
record = records.get("wekws-dstcn")
model = record.fetch("a8w8", root=Path("/path/to/selected-model-data"))
```

Cards have one canonical repository location, `models/<id>/README.md`, matching record@1’s existing card resolver. `record.card(root=checkout)` reads the card normally; supplied data roots also retain that same card layout. These card-only directories do not add records to the default catalog. Load the `intake-records` parent, not an individual model directory.

The supplied data root has the existing `models/<id>/<precision>/model.tflite` layout. The caller binds graph and full diagnostic NPZ with standard Artifact hashes, sizes and roots. A missing local payload is a source gap, not an available remote artifact. An original-publisher HTTPS URI stays checksum/size pinned and is fetched by the same normal resolver. There is no alternate downloader or catalog.

Schema/read/fetch success does not override preparation gates. NNID retains its actual feedback zero-point mismatch; E2A2 retains its existing consumer refusal for the unbound output quantizer/full fixture; the two FP32-compute/FP16-storage records retain the actual native-half precision-policy refusal. Keep these outcomes visible. No validator rule is weakened and no fake golden is supplied.

| Source-only record | Card and exact IO/source limits |
| --- | --- |
| `acdnet20-esc50` | [Record](acdnet20-esc50/record.json), [source card](../models/acdnet20-esc50/README.md) |
| `cp-mobile-dcase2025` | [Record](cp-mobile-dcase2025/record.json), [source card](../models/cp-mobile-dcase2025/README.md) |
| `e2a2-clean-speech-op4` | [Record](e2a2-clean-speech-op4/record.json), [source card](../models/e2a2-clean-speech-op4/README.md) |
| `efficientat-mn02` | [Record](efficientat-mn02/record.json), [source card](../models/efficientat-mn02/README.md) |
| `heartkit-arr-fp32-w16` | [Record](heartkit-arr-fp32-w16/record.json), [source card](../models/heartkit-arr-fp32-w16/README.md) |
| `heartkit-seg-fp32-w16` | [Record](heartkit-seg-fp32-w16/record.json), [source card](../models/heartkit-seg-fp32-w16/README.md) |
| `mcunet-in1` | [Record](mcunet-in1/record.json), [source card](../models/mcunet-in1/README.md) |
| `microdpad-derived-fusion` | [Record](microdpad-derived-fusion/record.json), [source card](../models/microdpad-derived-fusion/README.md) |
| `mixtoken-relu-imu` | [Record](mixtoken-relu-imu/record.json), [source card](../models/mixtoken-relu-imu/README.md) |
| `mobilenetv2-035-96` | [Record](mobilenetv2-035-96/record.json), [source card](../models/mobilenetv2-035-96/README.md) |
| `mobileone-s0-native96` | [Record](mobileone-s0-native96/record.json), [source card](../models/mobileone-s0-native96/README.md) |
| `nnid-tflm` | [Record](nnid-tflm/record.json), [source card](../models/nnid-tflm/README.md) |
| `sleepkit-apnea` | [Record](sleepkit-apnea/record.json), [source card](../models/sleepkit-apnea/README.md) |
| `speech-to-intent-micro` | [Record](speech-to-intent-micro/record.json), [source card](../models/speech-to-intent-micro/README.md) |
| `wekws-dstcn` | [Record](wekws-dstcn/record.json), [source card](../models/wekws-dstcn/README.md) |
| `yamnet256-fsd5` | [Record](yamnet256-fsd5/record.json), [source card](../models/yamnet256-fsd5/README.md) |
