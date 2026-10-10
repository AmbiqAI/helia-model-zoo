# compressionKIT PPG 4x encoder

Selected fixed-batch trained A8W8 graph: 28,944 bytes, SHA256 `e570dc9a4b43df106c62983612fbe59ff1d39a31dc296b18c09411d07586cdae`. Complete stateless INT8 IO is `[1,1,320,1]` to `[1,1,80,16]`, with exact names and quantizers in `record.json`. This is the encoder only; codebook quantization, decoder and full codec/DSP are separate.

## Modified source and training

The [publisher](https://huggingface.co/Ambiq/compressionkit-ppg-4x-v1.0/tree/8e3b465983808d50d99799df6135aaffe68029ff) distributes a trained codec. Original `encoder_int8.tflite` is 28,928 bytes, SHA256 `5f1e0e62be756a3f486609fd98a8cf18d491187b76f8158e1394a1186981274b`, with dynamic batch signatures. The selected graph is a **modified version**: every batch signature was fixed to one by the retained preparation; weights, quantizers and operators were preserved. It is not the publisher's byte-identical dynamic file. No new conversion, calibration or parity test is represented here. The retained publisher model card and quantization report declare upstream training/calibration; downstream task/numerical qualification is not asserted.

The existing benchmark companion is diagnostic full-IO allocation, not a reference golden or redistributed sample dataset. No golden is supplied. Other precisions, including an INT16x8 graph with FLOAT32 IO, remain separate.

## Tier, retrieval and rights

Tier: selected graph and complete IO metadata prepared for the existing repository Git LFS route. Local payload is present; hosted retrieval is pending explicit publication/artifact transfer, so no available public commit reference is advertised. The stable dynamic upstream file stays at its publisher, but does not retrieve this fixed-batch graph.

Weights and modified quantized artifacts use [the retained Ambiq Model Weights License](LICENSE-MODEL-WEIGHTS.md), SHA256 `d2d5715e0b37f06a7d7b0f2638b05079d94e6815700a74f4019529651d5f8671`. It permits Ambiq-silicon production deployment, evaluation/benchmarking elsewhere for eventual Ambiq use, and redistribution with its notice and modified-version marking. Public results require attribution to Ambiq Micro, Inc. and the originating Hugging Face repository. These weight terms are separate from toolkit code and dataset terms.
