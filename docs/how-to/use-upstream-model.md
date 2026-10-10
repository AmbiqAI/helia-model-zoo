# Use an upstream model without mirroring it

Keep a stable publisher artifact at its original host. A normal
`helia-model-zoo/record@1` precision can use a direct HTTPS `uri` instead of a
repository `path`. Its `sha256` and positive `bytes` are required. The URL must
have a host and file path, with no credentials, query, fragment or dot-led path
component. A pinned commit in the publisher URL identifies its provenance;
the hash identifies the exact bytes even when the host's URL is mutable.

Keep the normal ordered `inputs` and `outputs` descriptors, their quantizers,
and `io.streaming`/`state_pairs`. Record the upstream revision and graph hash,
applicable model licence, and card. Use `license.spdx: null` when weight terms
are unknown; a repository's code licence alone is not a model licence. An
anonymous fetch proves access and identity, not redistribution rights or task
quality. Public benchmark inclusion and hosted redistribution require their
own eligibility decision. Private or internal paths stay in caller-supplied
metadata, outside this public repository.

Read a supplied record and fetch its model through the existing client:

```python
import json
from pathlib import Path
import helia_model_zoo as zoo

metadata = json.loads(Path("record.json").read_text())
record = zoo.parse_record(metadata, metadata["id"])
model = record.fetch("a8w8", cache=Path("owned-cache"), anonymous=True)
precision = record.precision("a8w8")
```

A valid cached file is reused after size/hash checks; changed bytes are fetched
again and verified. A wrong download leaves no verified cache entry. The
repository revision option does not retarget an external HTTPS source: its URL
and content pin control retrieval. Existing `hf://` references continue to
require a full commit and use Hugging Face's normal authentication.

If a reference golden exists, retain its real runtime/resolver and source
metadata and fetch it with `record.golden()`. Otherwise omit `golden` rather
than declaring an output allocation to be a reference. Save complete input
initialization and all output extents separately in the benchmark fixture;
stateful fixtures must supply every state input and define resets/carry
explicitly. A saved diagnostic input is not task calibration or evaluation.

A supplied record is not automatically adopted into the packaged zoo. A
`zoo://...@<commit>` reference is usable only when that record actually exists
at the named public commit or pinned private overlay. Local artifacts that
have no stable upstream graph remain explicit supplied-local dependencies;
do not substitute another family member or create an unresolvable reference.
