# compressionKIT ECG categorical prior

Original publisher graph SHA256 `1b90f6a5d43cd6e1714d558313ec21c1cbc2f86e16f52d220040ab44f09ecefa`,240280bytes, at immutable Hugging Face revision548a82529fc10075c31f19ab1ed8c09dfb2e58f4. Normal anonymous checksum/size-verified retrieval succeeded; no repository mirror is supplied. Resolve the source record at its exact repository revision; the graph itself remains at the checksum-pinned original publisher.

## Precision and interface

The `a8w8` key selects the publisher's prior_int8 artifact, not a pure-integer compute claim: the graph contains147INT8 and10FLOAT32 tensors, plus integer indices and Boolean masks. Input is unquantized INT32 token IDs[1,256],1024bytes; output is INT8 logits[1,256,256],65536bytes with its recorded quantizer. Zero variable tensors; complete256-token context is retained. No shortened prior, decoder, waveform frontend or end-to-end codec is represented.

## Diagnostic initialization

For a fixed diagnostic case, initialize all256token IDs to integer0. Restore the whole input eachInvoke and capture all65536output bytes. Rawzero output allocation is not a reference golden or generation-quality result; no golden field is supplied. Autoregressive sampling/task/codebook conventions remain separate and unqualified.

## Rights and training

Exact source is publisher-distributed quantized weights; separate original trained checkpoint/configuration and applicable model-weight redistribution terms are not bound here. License SPDX remains null. Unknown terms block mirroring/public benchmark admission, not checksum-bound retrieval from the original publisher. Do not infer model-weight rights from toolkit code. No task/numerical/native/device result is inherited by this source record.
