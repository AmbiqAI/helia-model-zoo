# MobileNet V2 1.0 224

MobileNet V2 image classification model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | standalone |
| Task | image classification |
| Model artifact | `vision/mobilenet_v2/model.tflite` |
| Golden fixture | `vision/mobilenet_v2/golden.npz` |
| Inputs | 1 tensor, `[1, 224, 224, 3]`, `int8` |
| Outputs | 1 tensor, `[1, 1001]`, `int8` |
| Precision | int8 |
| Provenance | Prebuilt MobileNet V2 TFLite artifact checked into this repo |
| Notes | Includes a checked-in golden fixture in the model directory |
