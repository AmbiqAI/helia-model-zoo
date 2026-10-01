# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Manifest v2 (``helia-model-zoo/manifest@2``): parsing and structural checks.

Standard library only, so consumers without NumPy can read the manifest.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .golden import GoldenData

SCHEMA = "helia-model-zoo/manifest@2"

PRECISIONS = ("fp32", "fp16", "int8", "int16x8")
TIERS = ("native", "converted")
VISIBILITIES = ("public", "private")
STREAMING = ("stateless", "explicit_state", "internal_state")
GOLDEN_KINDS = ("single", "batch", "sequence")
RESOLVERS = ("builtin", "builtin_ref")
DTYPES = ("float32", "float16", "int8", "uint8", "int16", "int32", "int64", "bool")
# lfs:// is a Git LFS file in this repository, repo:// a plain file in it; https:// is any HTTPS URL;
# hf://[datasets/|spaces/]<org>/<repo>@<40-hex commit>/<path> is a Hugging Face file (HfFileSystem form).
ARTIFACT_SCHEMES = ("lfs", "https", "hf")
TEXT_SCHEMES = ("repo", "https", "hf")

_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_URI = re.compile(r"(?P<scheme>[a-z]+)://(?P<path>.+)")
_HF = re.compile(
    r"(?:(?P<kind>datasets|spaces)/)?(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)@(?P<revision>[0-9a-f]{40})/(?P<path>.+)"
)
_HOST = re.compile(r"[A-Za-z0-9.-]+(:[0-9]+)?/.+")


@dataclass(frozen=True)
class HfLocation:
    """Where an ``hf://`` URI points: ``repo_type`` is ``model``, ``dataset`` or ``space``."""

    repo_type: str
    repo_id: str
    revision: str
    path: str


def parse_hf(uri: str) -> HfLocation:
    """Split an ``hf://`` URI; the revision must be a full 40-hex commit.

    Raises:
        ValueError: If ``uri`` is not of that form.
    """
    match = _HF.fullmatch(uri.removeprefix("hf://")) if uri.startswith("hf://") else None
    if not match:
        raise ValueError(f"expected hf://[datasets/|spaces/]<org>/<repo>@<40-hex commit>/<path>, got {uri!r}")
    kind = {"datasets": "dataset", "spaces": "space", None: "model"}[match["kind"]]
    return HfLocation(kind, match["repo"], match["revision"], match["path"])


class ManifestError(ValueError):
    """The manifest does not follow manifest@2."""


@dataclass(frozen=True)
class FileRef:
    """A file named by URI, pinned by sha256 when it is an artifact."""

    uri: str
    sha256: str | None = None
    bytes: int | None = None

    @property
    def scheme(self) -> str:
        return self.uri.split("://", 1)[0]

    @property
    def path(self) -> str:
        return self.uri.split("://", 1)[1]

    @property
    def in_repository(self) -> bool:
        """Whether this file lives in this repository (``lfs://`` or ``repo://``)."""
        return self.scheme in ("lfs", "repo")

    def resolve(self, root: Path) -> Path:
        """This file's path in a checkout of this repository.

        Raises:
            ValueError: If the file is not in this repository, or its path leaves ``root``
                (for example through a symlink).
        """
        if not self.in_repository:
            raise ValueError(f"{self.uri} is not a file in this repository")
        root = Path(root).resolve()
        candidate = (root / self.path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"{self.uri} escapes the repository root")
        return candidate


@dataclass(frozen=True)
class Tensor:
    """A model input or output as LiteRT reports it; ``scale`` is None when unquantized."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    scale: float | None
    zero_point: int | None


@dataclass(frozen=True)
class StatePair:
    """An explicit state: ``outputs[output]`` feeds ``inputs[input]`` on the next call.

    ``reset`` is the value that starts a stream: ``"zeros"`` is real-valued zero, which a
    quantized tensor stores as its zero point. ``scales_tied`` is true when the input and
    output share a scale, so the output can be copied to the input unchanged.
    """

    input: int
    output: int
    reset: str
    scales_tied: bool


@dataclass(frozen=True)
class IO:
    streaming: str
    state_pairs: tuple[StatePair, ...]
    sample_rate_hz: int | None = None
    frame_samples: int | None = None
    hop_samples: int | None = None
    context_samples: int | None = None


@dataclass(frozen=True)
class Golden:
    """A golden NPZ: ``input_i``/``output_i`` in model I/O order; batch and sequence add a leading step axis."""

    file: FileRef
    kind: str
    steps: int
    resets: tuple[int, ...]
    source: dict[str, Any] | None
    reference_runtime: str
    reference_runtime_version: str
    resolver: str


@dataclass(frozen=True)
class Precision:
    name: str
    model: FileRef
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    golden: Golden | None


@dataclass(frozen=True)
class License:
    spdx: str | None
    reference: FileRef


@dataclass(frozen=True)
class Upstream:
    repo: str
    revision: str | None
    path: str | None
    sha256: str | None


@dataclass(frozen=True)
class Entry:
    id: str
    aliases: dict[str, str]
    title: str
    domain: str
    task: str
    tier: str
    visibility: str
    license: License
    card_file: FileRef
    upstream: Upstream | None
    io: IO
    precisions: dict[str, Precision]

    def precision(self, name: str | None = None) -> Precision:
        """The named precision, or the only one when ``name`` is None."""
        if name is None:
            if len(self.precisions) != 1:
                raise KeyError(f"{self.id} has precisions {sorted(self.precisions)}; name one")
            return next(iter(self.precisions.values()))
        if name not in self.precisions:
            raise KeyError(f"{self.id} has no precision {name!r}; it has {sorted(self.precisions)}")
        return self.precisions[name]

    def fetch(self, precision: str | None = None, **options: Any) -> Path:
        """The verified local path of a precision's model; ``options`` go to ``fetch_file``."""
        from .hydrate import fetch_file

        return fetch_file(self.precision(precision).model, **options)

    def golden(self, precision: str | None = None, **options: Any) -> GoldenData:
        """A precision's golden arrays and metadata, from a verified local file."""
        from .golden import load_golden
        from .hydrate import fetch_file

        chosen = self.precision(precision)
        if chosen.golden is None:
            raise KeyError(f"{self.id} {chosen.name} has no golden")
        return load_golden(fetch_file(chosen.golden.file, **options), chosen)

    def card(self, **options: Any) -> Path:
        """The verified local path of the model card."""
        from .hydrate import fetch_file

        return fetch_file(self.card_file, **options)


@dataclass(frozen=True)
class Manifest:
    entries: tuple[Entry, ...]

    def get(self, model_id: str) -> Entry:
        """Return the entry with this ID or alias."""
        for entry in self.entries:
            if model_id == entry.id or model_id in entry.aliases:
                return entry
        raise KeyError(f"unknown model ID: {model_id}")

    def resolve(self, model_id: str) -> tuple[Entry, Precision | None]:
        """Return the entry for an ID or alias, and the precision an alias names."""
        entry = self.get(model_id)
        if model_id in entry.aliases:
            return entry, entry.precisions[entry.aliases[model_id]]
        return entry, None


def _object(value: Any, where: str, required: set[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where}: expected an object")
    missing = required - value.keys()
    if missing:
        raise ManifestError(f"{where}: missing {sorted(missing)}")
    unknown = value.keys() - required - optional
    if unknown:
        raise ManifestError(f"{where}: unknown keys {sorted(unknown)}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: expected a non-empty string")
    return value


def _choice(value: Any, where: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ManifestError(f"{where}: expected one of {list(choices)}, got {value!r}")
    return value


def _int(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{where}: expected an integer >= {minimum}")
    return value


def _optional_int(value: Any, where: str) -> int | None:
    return None if value is None else _int(value, where, 1)


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ManifestError(f"{where}: expected a lowercase hex sha256")
    return value


def _uri(value: Any, where: str, schemes: tuple[str, ...]) -> str:
    match = _URI.fullmatch(_string(value, where))
    if not match or match["scheme"] not in schemes:
        raise ManifestError(f"{where}: expected a URI with scheme {list(schemes)}, got {value!r}")
    path = match["path"]
    if match["scheme"] == "hf":
        try:
            path = parse_hf(value).path
        except ValueError as error:
            raise ManifestError(f"{where}: {error}") from None
    elif match["scheme"] == "https":
        if not _HOST.fullmatch(path):
            raise ManifestError(f"{where}: expected https://<host>/<path>, got {value!r}")
        return value
    if path.startswith("/") or ".." in Path(path).parts:
        raise ManifestError(f"{where}: path must not be absolute or contain '..': {path!r}")
    return value


def _file(value: Any, where: str, artifact: bool) -> FileRef:
    if artifact:
        data = _object(value, where, {"uri", "sha256", "bytes"})
        return FileRef(
            _uri(data["uri"], f"{where}.uri", ARTIFACT_SCHEMES),
            _sha256(data["sha256"], f"{where}.sha256"),
            _int(data["bytes"], f"{where}.bytes", 1),
        )
    data = _object(value, where, {"uri"}, frozenset({"sha256"}))
    uri = _uri(data["uri"], f"{where}.uri", TEXT_SCHEMES)
    sha256 = None if data.get("sha256") is None else _sha256(data["sha256"], f"{where}.sha256")
    if uri.startswith("https://") and sha256 is None:
        raise ManifestError(f"{where}: an https:// file needs a sha256, since its URL is not pinned to a revision")
    return FileRef(uri, sha256)


def _tensor(value: Any, where: str) -> Tensor:
    data = _object(value, where, {"name", "shape", "dtype", "scale", "zero_point"})
    shape = data["shape"]
    if not isinstance(shape, list) or not all(type(d) is int and d >= -1 for d in shape):
        raise ManifestError(f"{where}.shape: expected a list of integers (-1 for dynamic)")
    scale, zero_point = data["scale"], data["zero_point"]
    if (scale is None) != (zero_point is None):
        raise ManifestError(f"{where}: scale and zero_point must both be set or both be null")
    if scale is not None and (
        type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0 or type(zero_point) is not int
    ):
        raise ManifestError(f"{where}: expected a positive scale and an integer zero_point")
    return Tensor(
        _string(data["name"], f"{where}.name"),
        tuple(shape),
        _choice(data["dtype"], f"{where}.dtype", DTYPES),
        None if scale is None else float(scale),
        zero_point,
    )


def _golden(value: Any, where: str) -> Golden:
    keys = {"file", "kind", "steps", "resets", "source", "reference_runtime", "reference_runtime_version", "resolver"}
    data = _object(value, where, keys)
    kind = _choice(data["kind"], f"{where}.kind", GOLDEN_KINDS)
    steps = _int(data["steps"], f"{where}.steps", 1)
    if kind == "single" and steps != 1:
        raise ManifestError(f"{where}: a single golden has exactly one step")
    resets = data["resets"]
    if not isinstance(resets, list) or not all(type(t) is int and 0 < t < steps for t in resets):
        raise ManifestError(f"{where}.resets: expected step indices in [1, steps)")
    if kind != "sequence" and resets:
        raise ManifestError(f"{where}.resets: only a sequence golden has resets")
    source = data["source"]
    if source is not None:
        if isinstance(source, dict) and source.keys() == {"seed"} and type(source["seed"]) is int:
            pass
        elif isinstance(source, dict) and source.keys() == {"uri", "sha256"}:
            _string(source["uri"], f"{where}.source.uri")
            _sha256(source["sha256"], f"{where}.source.sha256")
        else:
            raise ManifestError(f"{where}.source: expected null, {{seed}} or {{uri, sha256}}")
    return Golden(
        _file(data["file"], f"{where}.file", artifact=True),
        kind,
        steps,
        tuple(resets),
        source,
        _string(data["reference_runtime"], f"{where}.reference_runtime"),
        _string(data["reference_runtime_version"], f"{where}.reference_runtime_version"),
        _choice(data["resolver"], f"{where}.resolver", RESOLVERS),
    )


def _precision(name: str, value: Any, where: str) -> Precision:
    data = _object(value, where, {"model", "inputs", "outputs"}, frozenset({"golden"}))
    tensors = {}
    for role in ("inputs", "outputs"):
        if not isinstance(data[role], list) or not data[role]:
            raise ManifestError(f"{where}.{role}: expected a non-empty list")
        tensors[role] = tuple(_tensor(t, f"{where}.{role}[{i}]") for i, t in enumerate(data[role]))
    golden = data.get("golden")
    return Precision(
        name,
        _file(data["model"], f"{where}.model", artifact=True),
        tensors["inputs"],
        tensors["outputs"],
        None if golden is None else _golden(golden, f"{where}.golden"),
    )


def _io(value: Any, where: str) -> IO:
    optional = frozenset({"sample_rate_hz", "frame_samples", "hop_samples", "context_samples"})
    data = _object(value, where, {"streaming", "state_pairs"}, optional)
    streaming = _choice(data["streaming"], f"{where}.streaming", STREAMING)
    if not isinstance(data["state_pairs"], list):
        raise ManifestError(f"{where}.state_pairs: expected a list")
    pairs = []
    for i, pair in enumerate(data["state_pairs"]):
        item = _object(pair, f"{where}.state_pairs[{i}]", {"input", "output", "reset", "scales_tied"})
        if type(item["scales_tied"]) is not bool:
            raise ManifestError(f"{where}.state_pairs[{i}].scales_tied: expected a boolean")
        pairs.append(
            StatePair(
                _int(item["input"], f"{where}.state_pairs[{i}].input"),
                _int(item["output"], f"{where}.state_pairs[{i}].output"),
                _choice(item["reset"], f"{where}.state_pairs[{i}].reset", ("zeros",)),
                item["scales_tied"],
            )
        )
    if (streaming == "explicit_state") != bool(pairs):
        raise ManifestError(f"{where}: explicit_state needs state_pairs, and only explicit_state has them")
    for role in ("input", "output"):
        used = [getattr(p, role) for p in pairs]
        if len(set(used)) != len(used):
            raise ManifestError(f"{where}.state_pairs: an {role} is paired twice")
    return IO(streaming, tuple(pairs), **{k: _optional_int(data.get(k), f"{where}.{k}") for k in optional})


def _entry(value: Any, where: str) -> Entry:
    keys = {
        "id",
        "aliases",
        "title",
        "domain",
        "task",
        "tier",
        "visibility",
        "license",
        "card",
        "upstream",
        "io",
        "precisions",
    }
    data = _object(value, where, keys)
    entry_id = _string(data["id"], f"{where}.id")
    if not _ID.fullmatch(entry_id):
        raise ManifestError(f"{where}.id: expected lowercase letters, digits, '.' and '-': {entry_id!r}")
    where = f"entry {entry_id}"
    if not isinstance(data["precisions"], dict) or not data["precisions"]:
        raise ManifestError(f"{where}.precisions: expected a non-empty object")
    precisions = {
        _choice(k, f"{where}.precisions", PRECISIONS): _precision(k, v, f"{where}.precisions.{k}")
        for k, v in data["precisions"].items()
    }
    aliases = data["aliases"]
    if not isinstance(aliases, dict) or not all(
        isinstance(k, str) and _ID.fullmatch(k) and isinstance(v, str) for k, v in aliases.items()
    ):
        raise ManifestError(f"{where}.aliases: expected an object of ID -> precision")
    for alias, precision in aliases.items():
        if precision not in precisions:
            raise ManifestError(f"{where}.aliases.{alias}: names a missing precision {precision!r}")
    lic = _object(data["license"], f"{where}.license", {"spdx", "reference"})
    spdx = None if lic["spdx"] is None else _string(lic["spdx"], f"{where}.license.spdx")
    upstream = None
    if data["upstream"] is not None:
        up = _object(data["upstream"], f"{where}.upstream", {"repo", "revision", "path", "sha256"})
        revision = up["revision"]
        if revision is not None and not (isinstance(revision, str) and _REVISION.fullmatch(revision)):
            raise ManifestError(f"{where}.upstream.revision: expected a full 40-hex commit or null")
        upstream = Upstream(
            _string(up["repo"], f"{where}.upstream.repo"),
            revision,
            None if up["path"] is None else _string(up["path"], f"{where}.upstream.path"),
            None if up["sha256"] is None else _sha256(up["sha256"], f"{where}.upstream.sha256"),
        )
    io = _io(data["io"], f"{where}.io")
    for precision in precisions.values():
        for pair in io.state_pairs:
            if pair.input >= len(precision.inputs) or pair.output >= len(precision.outputs):
                raise ManifestError(f"{where}.io.state_pairs: index out of range for {precision.name}")
    return Entry(
        entry_id,
        dict(aliases),
        _string(data["title"], f"{where}.title"),
        _string(data["domain"], f"{where}.domain"),
        _string(data["task"], f"{where}.task"),
        _choice(data["tier"], f"{where}.tier", TIERS),
        _choice(data["visibility"], f"{where}.visibility", VISIBILITIES),
        License(spdx, _file(lic["reference"], f"{where}.license.reference", artifact=False)),
        _file(data["card"], f"{where}.card", artifact=False),
        upstream,
        io,
        precisions,
    )


def merge(*manifests: Manifest) -> Manifest:
    """One manifest holding every entry; IDs and aliases must stay unique across all of them.

    Raises:
        ManifestError: If an ID or alias is used twice.
    """
    entries = tuple(entry for manifest in manifests for entry in manifest.entries)
    _check_unique(entries)
    return Manifest(entries)


def _check_unique(entries: tuple[Entry, ...]) -> None:
    names: dict[str, str] = {}
    for entry in entries:
        for name in (entry.id, *entry.aliases):
            if name in names:
                raise ManifestError(f"entry {entry.id}: ID or alias {name!r} is already used by {names[name]}")
            names[name] = entry.id


def parse_manifest(data: Any) -> Manifest:
    """Check a decoded manifest@2 document and return it as typed records.

    Raises:
        ManifestError: On the first structural problem found.
    """
    data = _object(data, "manifest", {"schema", "entries"})
    if data["schema"] != SCHEMA:
        raise ManifestError(f"manifest: unsupported schema {data['schema']!r}; expected {SCHEMA!r}")
    if not isinstance(data["entries"], list) or not data["entries"]:
        raise ManifestError("manifest: expected at least one entry")
    entries = tuple(_entry(value, f"entries[{i}]") for i, value in enumerate(data["entries"]))
    _check_unique(entries)
    return Manifest(entries)


def load_manifest(path: Path | None = None) -> Manifest:
    """Load a manifest@2 file; by default the one shipped with this package."""
    if path is None:
        text = resources.files(__package__).joinpath("manifest.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return parse_manifest(json.loads(text))
