# DTLN noise-suppression stage

Source-only record for the unchanged publisher A8W8 graph: 372,720 bytes, SHA256 `91e94316d1a4747dd3bddcf58423927e97213913a9b4210edca3f9402b2ff0fe`. The record pins tensorflow/tflite-micro commit `3f787dcff5c4693f3436a65aff1b268538a07bdb` and the DTLN example's `dtln_noise_suppression.tflite`.

The [publisher example](https://github.com/tensorflow/tflite-micro/blob/3f787dcff5c4693f3436a65aff1b268538a07bdb/tensorflow/lite/micro/examples/dtln/README.md) describes a model retrained by Cadence on DNS Challenge data, with the noise-suppression stage quantized to 8 bits. It excludes the paper's speech-enhancement stage and expressly does not support denoising-quality evaluation. This is a publisher training declaration, not independently verified checkpoint or task-quality evidence.

## Interface and state

The complete external interface is one INT8 `[1,1,257]` feature input and one INT8 `[1,1,257]` output. This is a recurrent stage, not a stateless full audio pipeline. Four mutable internal tensors retain two LSTM hidden/cell pairs, each `[1,128]`:

| Internal tensor | Dtype | Scale | Real-zero code |
| --- | --- | --- | ---: |
| `tfl.pseudo_qconst` | INT8 | 0.007654283195734024 | 0 |
| `tfl.pseudo_qconst1` | INT16 | 0.00048828125 | 0 |
| `tfl.pseudo_qconst2` | INT8 | 0.00736330496147275 | -4 |
| `tfl.pseudo_qconst3` | INT16 | 0.000244140625 | 0 |

Reset must honor all four quantized real-zero codes at sequence boundaries; memset-zero is insufficient. The A8W8 label names the shipped variant and does not hide its INT16 internal cell state. Frontend framing/cadence, sequence reset/continuity and the full saved state fixture remain unqualified. The existing internal-state preparation refusal and code-generation-only outcome remain separate from model retrieval. No runtime-support, target execution, timing or numerical acceptance follows from this record. No reference golden is supplied.

## Source tier and rights

Tier: pinned upstream stage and external IO descriptors; complete mutable-state fixture and reset/sequence qualification pending. Other precisions require a trained source reconstruction and are not supplied. Installed-package discovery and verified model retrieval are supported; a revision reference is usable only after publication at that revision.

The publisher repository declares [Apache-2.0](https://github.com/tensorflow/tflite-micro/blob/3f787dcff5c4693f3436a65aff1b268538a07bdb/LICENSE). A separate model-weight licence and original training checkpoint have not been established, so `license.spdx` remains null. This record publishes source metadata and links to the unchanged publisher artifact; it does not redistribute or mirror weights. Public benchmark recipe eligibility and managed hosting remain pending weight-rights disposition.
