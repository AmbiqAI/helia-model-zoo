# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Model records (``helia-model-zoo/record@1``): one ``models/<id>/record.json`` per model.

Standard library only, so consumers without NumPy can read the records.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .golden import GoldenData

SCHEMA = "helia-model-zoo/record@1"
MODELS = "models"

PRECISIONS = ("fp32", "fp16", "a8w8", "a16w8", "a8w4")
VISIBILITIES = ("public", "private")
STREAMING = ("stateless", "explicit_state", "internal_state")
GOLDEN_KINDS = ("single", "batch", "sequence")
RESOLVERS = ("builtin", "builtin_ref")
DTYPES = ("float32", "float16", "int8", "uint8", "int16", "int32", "int64", "bool")

_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_RUNTIME = re.compile(r"(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9.+_-]+)")
# A relative path: no empty or "."-led parts, backslashes, or characters below 0x20.
_PATH = re.compile(r"[^/\\\x00-\x1f.][^/\\\x00-\x1f]*(?:/[^/\\\x00-\x1f.][^/\\\x00-\x1f]*)*")
_HF = re.compile(
    r"(?:(?P<kind>datasets|spaces)/)?(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)@(?P<revision>[0-9a-f]{40})/(?P<path>.+)"
)


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
    if not match or not _PATH.fullmatch(match["path"]):
        raise ValueError(f"expected hf://[datasets/|spaces/]<org>/<repo>@<40-hex commit>/<path>, got {uri!r}")
    kind = {"datasets": "dataset", "spaces": "space", None: "model"}[match["kind"]]
    return HfLocation(kind, match["repo"], match["revision"], match["path"])


def https_path(uri: str) -> str:
    """Validate an anonymous HTTPS file URL and return its relative file path.

    Queries, fragments, credentials and dot-led path components are not supported.
    The file's content is pinned separately by its SHA-256 and byte size.
    """
    try:
        url = urlsplit(uri)
        valid = (
            uri.startswith("https://")
            and url.hostname
            and url.port != 0
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and not any(c.isspace() or ord(c) < 32 or c == "\\" for c in uri)
            and _PATH.fullmatch(url.path.removeprefix("/"))
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("expected an HTTPS file URL without credentials, query or fragment")
    return url.path.removeprefix("/")


class ManifestError(ValueError):
    """A record does not follow record@1."""


@dataclass(frozen=True)
class FileRef:
    """A file: ``lfs://`` or ``repo://`` (an artifact or a text file in this repository, by
    repository path), ``hf://`` (a Hugging Face file at a pinned revision), or
    ``https://`` (an upstream artifact pinned by hash and size)."""

    uri: str
    sha256: str | None = None
    bytes: int | None = None

    @property
    def scheme(self) -> str:
        return self.uri.split("://", 1)[0]

    @property
    def path(self) -> str:
        return https_path(self.uri) if self.scheme == "https" else self.uri.split("://", 1)[1]

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
    """An explicit state, by tensor name: the ``output`` of one call feeds the ``input`` of the next.

    A stream starts from real-valued zero, which a quantized tensor stores as its zero point.
    """

    input: str
    output: str


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

    def pair_indices(self, io: IO) -> tuple[tuple[int, int], ...]:
        """Each state pair as ``(input index, output index)`` in this precision's tensors.

        Raises:
            KeyError: If a paired tensor name is not one of this precision's tensors.
        """
        inputs = {t.name: i for i, t in enumerate(self.inputs)}
        outputs = {t.name: i for i, t in enumerate(self.outputs)}
        try:
            return tuple((inputs[p.input], outputs[p.output]) for p in io.state_pairs)
        except KeyError as error:
            raise KeyError(f"precision {self.name} has no tensor named {error.args[0]!r}") from None


@dataclass(frozen=True)
class Upstream:
    repo: str
    revision: str | None
    path: str | None
    sha256: str | None


@dataclass(frozen=True)
class Record:
    id: str
    title: str
    domain: str
    task: str
    visibility: str
    spdx: str | None
    card_file: FileRef
    upstream: Upstream | None
    io: IO
    precisions: dict[str, Precision]
    revision: str | None = None
    """The commit this record was read at by ``helia_model_zoo.get``: of this repository, or of the overlay
    dataset for a private record. None for the installed records."""

    def precision(self, name: str | None = None) -> Precision:
        """The named precision, or the only one when ``name`` is None."""
        if name is None:
            if len(self.precisions) != 1:
                raise KeyError(f"{self.id} has precisions {sorted(self.precisions)}; name one")
            return next(iter(self.precisions.values()))
        if name not in self.precisions:
            raise KeyError(f"{self.id} has no precision {name!r}; it has {sorted(self.precisions)}")
        return self.precisions[name]

    def fetch_file(self, ref: FileRef, **options: Any) -> Path:
        """The verified local path of one of this record's files, always at this record's revision when it has one.

        ``options`` go to ``helia_model_zoo.hydrate.fetch_file``.
        """
        from .hydrate import fetch_file

        if self.revision is not None:
            options["revision"] = self.revision
        return fetch_file(ref, **options)

    def fetch(self, precision: str | None = None, **options: Any) -> Path:
        """The verified local path of a precision's model."""
        return self.fetch_file(self.precision(precision).model, **options)

    def golden(self, precision: str | None = None, **options: Any) -> GoldenData:
        """A precision's golden arrays and metadata, from a verified local file."""
        from .golden import load_golden

        chosen = self.precision(precision)
        if chosen.golden is None:
            raise KeyError(f"{self.id} {chosen.name} has no golden")
        return load_golden(self.fetch_file(chosen.golden.file, **options), chosen)

    def files(self) -> tuple[FileRef, ...]:
        """The card, then each precision's model and golden."""
        files = [self.card_file]
        for precision in self.precisions.values():
            files.append(precision.model)
            if precision.golden is not None:
                files.append(precision.golden.file)
        return tuple(files)

    def card(self, **options: Any) -> Path:
        """The verified local path of the model card."""
        return self.fetch_file(self.card_file, **options)


@dataclass(frozen=True)
class Manifest:
    """Every record that a models directory (and any overlay) holds."""

    records: tuple[Record, ...]

    def get(self, model_id: str) -> Record:
        """The record with this ID."""
        for record in self.records:
            if record.id == model_id:
                return record
        raise KeyError(f"unknown model ID: {model_id}")


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


def _file(data: dict[str, Any], where: str, model_id: str, artifact: bool) -> FileRef:
    """A relative ``path`` or a pinned ``uri``, not both."""
    if ("path" in data) == ("uri" in data):
        raise ManifestError(f"{where}: give exactly one of path or uri")
    if "uri" in data:
        uri = _string(data["uri"], f"{where}.uri")
        try:
            if uri.startswith("https://") and artifact:
                https_path(uri)
            else:
                parse_hf(uri)
        except ValueError as error:
            raise ManifestError(f"{where}.uri: {error}") from None
    else:
        path = _string(data["path"], f"{where}.path")
        if not _PATH.fullmatch(path):
            raise ManifestError(f"{where}.path: expected a path inside the model directory, got {path!r}")
        uri = f"{'lfs' if artifact else 'repo'}://{MODELS}/{model_id}/{path}"
    if not artifact:
        return FileRef(uri)
    return FileRef(uri, _sha256(data["sha256"], f"{where}.sha256"), _int(data["bytes"], f"{where}.bytes", 1))


def _artifact(value: Any, where: str, model_id: str, extra: frozenset[str] = frozenset()) -> tuple[FileRef, dict]:
    data = _object(value, where, {"sha256", "bytes"}, frozenset({"path", "uri"}) | extra)
    return _file(data, where, model_id, artifact=True), data


def _tensor(value: Any, where: str) -> Tensor:
    data = _object(value, where, {"name", "shape", "dtype", "scale", "zero_point"})
    shape = data["shape"]
    if not isinstance(shape, list) or not all(type(d) is int and d >= -1 for d in shape):
        raise ManifestError(f"{where}.shape: expected a list of integers (-1 for dynamic)")
    scale, zero_point = data["scale"], data["zero_point"]
    if (scale is None) != (zero_point is None):
        raise ManifestError(f"{where}: scale and zero_point must both be set or both be null")
    if scale is not None and (
        type(scale) is not float or not math.isfinite(scale) or scale <= 0 or type(zero_point) is not int
    ):
        raise ManifestError(f"{where}: expected a positive float scale and an integer zero_point")
    return Tensor(
        _string(data["name"], f"{where}.name"),
        tuple(shape),
        _choice(data["dtype"], f"{where}.dtype", DTYPES),
        None if scale is None else float(scale),
        zero_point,
    )


def _golden(value: Any, where: str, model_id: str) -> Golden:
    keys = frozenset({"kind", "steps", "resets", "source", "runtime", "resolver"})
    file, data = _artifact(value, where, model_id, keys)
    missing = keys - data.keys()
    if missing:
        raise ManifestError(f"{where}: missing {sorted(missing)}")
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
    runtime = _RUNTIME.fullmatch(_string(data["runtime"], f"{where}.runtime"))
    if runtime is None:
        raise ManifestError(f"{where}.runtime: expected <distribution>==<version>, got {data['runtime']!r}")
    return Golden(
        file,
        kind,
        steps,
        tuple(resets),
        source,
        runtime["name"],
        runtime["version"],
        _choice(data["resolver"], f"{where}.resolver", RESOLVERS),
    )


def _precision(name: str, value: Any, where: str, model_id: str) -> Precision:
    data = _object(value, where, {"model", "inputs", "outputs"}, frozenset({"golden"}))
    tensors = {}
    for role in ("inputs", "outputs"):
        if not isinstance(data[role], list) or not data[role]:
            raise ManifestError(f"{where}.{role}: expected a non-empty list")
        tensors[role] = tuple(_tensor(t, f"{where}.{role}[{i}]") for i, t in enumerate(data[role]))
        names = [t.name for t in tensors[role]]
        if len(set(names)) != len(names):
            raise ManifestError(f"{where}.{role}: tensor names must be unique")
    model, _ = _artifact(data["model"], f"{where}.model", model_id)
    golden = data.get("golden")
    return Precision(
        name,
        model,
        tensors["inputs"],
        tensors["outputs"],
        None if golden is None else _golden(golden, f"{where}.golden", model_id),
    )


def _io(value: Any, where: str) -> IO:
    optional = frozenset({"sample_rate_hz", "frame_samples", "hop_samples", "context_samples"})
    data = _object(value, where, {"streaming", "state_pairs"}, optional)
    streaming = _choice(data["streaming"], f"{where}.streaming", STREAMING)
    if not isinstance(data["state_pairs"], list):
        raise ManifestError(f"{where}.state_pairs: expected a list")
    pairs = []
    for i, pair in enumerate(data["state_pairs"]):
        item = _object(pair, f"{where}.state_pairs[{i}]", {"in", "out"})
        pairs.append(
            StatePair(
                _string(item["in"], f"{where}.state_pairs[{i}].in"),
                _string(item["out"], f"{where}.state_pairs[{i}].out"),
            )
        )
    if (streaming == "explicit_state") != bool(pairs):
        raise ManifestError(f"{where}: explicit_state needs state_pairs, and only explicit_state has them")
    for role in ("input", "output"):
        used = [getattr(p, role) for p in pairs]
        if len(set(used)) != len(used):
            raise ManifestError(f"{where}.state_pairs: an {role} is paired twice")
    return IO(streaming, tuple(pairs), **{k: _optional_int(data.get(k), f"{where}.{k}") for k in optional})


def parse_record(data: Any, model_id: str) -> Record:
    """Check one decoded record@1 document for the model directory ``models/<model_id>/``.

    Raises:
        ManifestError: On the first structural problem found.
    """
    where = f"record {model_id}"
    keys = {"schema", "id", "title", "task", "domain", "visibility", "license", "card", "io", "precisions"}
    data = _object(data, where, keys, frozenset({"upstream"}))
    if data["schema"] != SCHEMA:
        raise ManifestError(f"{where}: unsupported schema {data['schema']!r}; expected {SCHEMA!r}")
    if data["id"] != model_id or not _ID.fullmatch(model_id):
        raise ManifestError(f"{where}: id must equal its directory name and use lowercase letters, digits, '.', '-'")
    if not isinstance(data["precisions"], dict) or not data["precisions"]:
        raise ManifestError(f"{where}.precisions: expected a non-empty object")
    precisions = {
        _choice(k, f"{where}.precisions", PRECISIONS): _precision(k, v, f"{where}.precisions.{k}", model_id)
        for k, v in data["precisions"].items()
    }
    lic = _object(data["license"], f"{where}.license", {"spdx"})
    spdx = None if lic["spdx"] is None else _string(lic["spdx"], f"{where}.license.spdx")
    card = data["card"]
    card_file = _file(
        {"uri": card} if isinstance(card, str) and card.startswith("hf://") else {"path": card},
        f"{where}.card",
        model_id,
        artifact=False,
    )
    upstream = None
    if data.get("upstream") is not None:
        up = _object(data["upstream"], f"{where}.upstream", {"repo"}, frozenset({"revision", "path", "sha256"}))
        revision = up.get("revision")
        if revision is not None and not (isinstance(revision, str) and _REVISION.fullmatch(revision)):
            raise ManifestError(f"{where}.upstream.revision: expected a full 40-hex commit")
        upstream = Upstream(
            _string(up["repo"], f"{where}.upstream.repo"),
            revision,
            None if up.get("path") is None else _string(up["path"], f"{where}.upstream.path"),
            None if up.get("sha256") is None else _sha256(up["sha256"], f"{where}.upstream.sha256"),
        )
    io = _io(data["io"], f"{where}.io")
    for precision in precisions.values():
        try:
            precision.pair_indices(io)
        except KeyError as error:
            raise ManifestError(f"{where}.io.state_pairs: {error.args[0]}") from None
    return Record(
        model_id,
        _string(data["title"], f"{where}.title"),
        _string(data["domain"], f"{where}.domain"),
        _string(data["task"], f"{where}.task"),
        _choice(data["visibility"], f"{where}.visibility", VISIBILITIES),
        spdx,
        card_file,
        upstream,
        io,
        precisions,
    )


def parse_records(records: dict[str, Any]) -> Manifest:
    """Records from decoded documents keyed by model ID (directory name)."""
    return Manifest(tuple(parse_record(data, model_id) for model_id, data in sorted(records.items())))


def merge(*manifests: Manifest) -> Manifest:
    """One manifest holding every record; IDs must stay unique across all of them.

    Raises:
        ManifestError: If an ID is used twice.
    """
    records = tuple(record for manifest in manifests for record in manifest.records)
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ManifestError(f"record {record.id}: ID is already used")
        seen.add(record.id)
    return Manifest(records)


def packaged_models() -> Path:
    """The models directory this package reads: its own data in a wheel, else the checkout's."""
    shipped = resources.files(__package__).joinpath(MODELS)
    if shipped.is_dir():
        return Path(str(shipped))
    return Path(__file__).resolve().parents[2] / MODELS


def read_record(content: bytes, where: object) -> Any:
    """Decode one ``record.json``.

    Raises:
        ManifestError: If it is not UTF-8 JSON.
    """
    try:
        return json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise ManifestError(f"{where}: {error}") from None


def load_manifest(models: Path | None = None) -> Manifest:
    """Load every ``<models>/<id>/record.json``; by default the records shipped with this package."""
    models = packaged_models() if models is None else Path(models)
    documents = {path.parent.name: read_record(path.read_bytes(), path) for path in models.glob("*/record.json")}
    if not documents:
        raise ManifestError(f"no records under {models}")
    return parse_records(documents)
