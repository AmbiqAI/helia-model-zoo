# Reproduce supplied model sources

Install the Zoo client from the exact repository revision required by the consumer. A wheel built from a local checkout does not carry a Git revision: pass the public record revision explicitly to `get()`/`resolve()`, or set `HELIA_ZOO_REVISION`. Set `HELIA_ZOO_CACHE` to a caller-owned directory. Hugging Face uses its separate `HF_HOME`/`HF_HUB_CACHE` and normal account authentication; do not copy credentials into a cache or recipe.

For public records, the normal source path is:

```python
from pathlib import Path
import helia_model_zoo as zoo

source = zoo.resolve(
    "zoo://mlperf-tiny-kws/a8w8@10cf244c84a0edbe84d44567208577f4a75349cd",
    cache=Path("source-cache"),
    anonymous=True,
)
```

This retrieves and verifies files; it does not invoke a model. A private selection needs the caller's existing pinned overlay and an account allowed to read both its records and model repository. Set `HELIA_ZOO_OVERLAY` before first catalog access, or call `zoo.manifest.cache_clear()` after changing it. Use `load_overlay(location, cache=cache)` for an explicit metadata-only read. See [portfolio sources](reproduce-portfolio-sources.md) for the existing Artifact/LocalSource interface and full diagnostic IO. Source availability, weight rights and engine admission are separate facts.

## Exact embedded SoundKit bytes

Two existing EVB headers contain literal `mut_model` arrays. The original headers are tracked at SoundKit revision `4530b04c0f2356bb3b78fa1274bc998319e1b482`; the SHA256/size pins below identify their actual bytes. A fresh station must obtain these exact headers through its authorized SoundKit checkout or as supplied sources. This guide does not establish anonymous remote access or weight redistribution rights.

| Source relative to the SoundKit root | Header bytes | Header SHA256 | Extracted model bytes | Model SHA256 |
|---|---:|---|---:|---|
| `soundkit/tasks/kws/evb_webusb/src/mut_model_data.h` | 218411 | `0fd18cba97813c2907665179a05a906e1b069c4e8fda906beccd10196459d857` | 34912 | `fff9759c2729f5a998edbf03b3b764484a494299dbebea1f71c5f75ac47bf4b0` |
| `soundkit/tasks/id/evb_webusb/src/mut_model_data.h` | 862063 | `95949d1c8cf79db4a5b4f75796f555741606588f87e868f20fa00b2341b76ea3` | 137896 | `58c51279492d49543eff9e2e5165580e9140bdcb6ce0ee781e69c8eeab03b0e2` |

With the first header supplied as `kws-model.h`, extract it through the installed command:

```sh
helia-zoo extract-header \
  --header kws-model.h \
  --header-sha256 0fd18cba97813c2907665179a05a906e1b069c4e8fda906beccd10196459d857 \
  --header-bytes 218411 --symbol mut_model \
  --sha256 fff9759c2729f5a998edbf03b3b764484a494299dbebea1f71c5f75ac47bf4b0 \
  --bytes 34912 --output kws-model.tflite
```

The command requires both identities, refuses a missing or changed header, nonliteral/ambiguous arrays, conditional definitions other than a canonical include guard, and wrong extracted bytes, and never overwrites an existing output. It copies literal bytes without loading an inference runtime, retraining or exporting a graph. The Python helper `helia_model_zoo.embedded.extract_c_array(header_bytes, symbol)` performs the literal extraction; callers of that helper must verify source and model identities separately.

The KWS graph has INT16 input `[1,4,40]` and output `[1,1,2]`; this differs from the current 257-bin `galaxy_focal` training configuration. Its exact training checkpoint is unknown. The speaker graph has INT16 input `[1,1,40]` and output `[1,1,64]`; the output is an embedding, and its exact relationship to the saved LibriSpeech checkpoint is unknown. Neither graph inherits checkpoint metrics from those training runs.

## Supplied raw graphs and complete IO

The two other SoundKit graphs are local result files, not tracked raw graphs at the header revision:

| Supplied source | Bytes | Graph SHA256 | External INT16 input/output |
|---|---:|---|---|
| `results/tasks/se/tflite/se_nn_arch72_mel_int16.tflite` | 120656 | `dfaffb59bc2d0b1e6977fad04f3df007e253d7923cd6ceadc06268e7f8572015` | `[1,1,72]` / `[1,1,257]` |
| `results/tasks/vad/tflite/crnn_int16.tflite` | 27424 | `ee1f842b8dacda7824704bb90d0886f4b3fdec72250b6ae843fe383b407583be` | `[1,1,257]` / `[1,1,2]` |

The original SE2 `se_v2_4.tflite` has SHA256 `7c364d71fb268847144f97729767d4c30e75812bc6d1a22927ea7f6411f50f4f`. Supply that exact graph; no remote selected source is established here. Its INT16 input and output are both `[1,1,129,2]`, 516 bytes each. A SoundKit speech-enhancement graph is not a substitute for it. Exact checkpoint/export parent linkage and weight terms are not inferred from a filename.

Keep the original graph's initializer subgraph and every resource-state literal. These models have fixed external IO and internal resource state, not extra external cache tensors. The saved source assessment binds zero-valued INT16 initializer literals with zero point 0; an input-only NPZ does not describe or replace those initializers. Initialization/state lifetime and supported execution are owned by the consuming engine.

Bind each selected graph and its complete diagnostic NPZ through the consumer's existing checksum-pinned Artifact/LocalSource fields, with caller-supplied roots. Preserve input/output order, dtype, shape and individual quantizers. Restore all saved inputs. Synthetic real-zero inputs use each tensor's zero point; output allocation arrays remain `NOT_REFERENCE_GOLDEN`. No timing, task, streaming or numerical qualification follows from successful extraction or source resolution. Missing original files remain explicit supplied-source prerequisites, rather than silently substituting another model or claiming remote portability.
