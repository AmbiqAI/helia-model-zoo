# GTCRN

GTCRN streaming speech enhancement model stored as a prebuilt int16 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech enhancement |
| Model artifact | `audio/gtcrn/model.tflite` |
| Golden fixture | Not generated |
| Inputs | 1 tensor, `[1, 129, 1, 3]`, `int16` |
| Outputs | 1 tensor, `[1, 129, 1, 2]`, `int16` |
| Precision | int16 |
| Provenance | Prebuilt GTCRN-derived TFLite artifact checked into this repo |
| Notes | No checked-in golden fixture is currently included for this model |
