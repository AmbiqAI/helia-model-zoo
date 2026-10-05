# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""The ``helia-zoo`` command line."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from .manifest import PRECISIONS, ManifestError, Record, load_manifest, parse_hf


def _manifest(args: argparse.Namespace):
    """``--models`` if given, else the packaged records plus any ``HELIA_ZOO_OVERLAY``."""
    if args.models is not None:
        return load_manifest(args.models)
    from . import manifest

    return manifest()


def inventory_row(record: Record) -> dict:
    """One model's inventory facts, all derived from its record."""
    upstream = record.upstream
    precisions = sorted(record.precisions, key=PRECISIONS.index)
    golden = [name for name in precisions if record.precisions[name].golden is not None]
    files = [ref for p in record.precisions.values() for ref in (p.model, *([p.golden.file] if p.golden else []))]
    licensed = record.spdx not in (None, "NOASSERTION", "NONE")
    gaps = [] if licensed else ["licence unknown"]
    if upstream is None or upstream.revision is None:
        gaps.append("source revision unknown")
    gaps += [f"no {name} golden" for name in precisions if name not in golden]
    return {
        "id": record.id,
        "title": record.title,
        "task": record.task,
        "domain": record.domain,
        "source": None if upstream is None else "@".join(filter(None, (upstream.repo, upstream.revision))),
        "license": record.spdx,
        "redistributable": "private" if record.visibility == "private" else "yes" if licensed else "unverified",
        "precisions": precisions,
        "golden": golden,
        "hosting": sorted({"git-lfs" if ref.in_repository else _hf_repo(ref.uri) for ref in files}),
        "gaps": gaps,
    }


def _hf_repo(uri: str) -> str:
    location = parse_hf(uri)
    kind = "" if location.repo_type == "model" else f"{location.repo_type}s/"
    return f"hf:{kind}{location.repo_id}"


def _cell(value) -> str:
    if value is None or value == []:
        return "-"
    text = ", ".join(value) if isinstance(value, list) else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _source(value: str | None) -> str | None:
    """A GitHub source as ``<org>/<repo>@<short revision>``."""
    if value is None:
        return None
    return re.sub(r"@([0-9a-f]{8})[0-9a-f]{32}$", r"@\1", value.removeprefix("https://github.com/"))


_COLUMNS = (
    ("ID", "id"),
    ("Task", "task"),
    ("Source", "source"),
    ("Licence", "license"),
    ("Redistributable", "redistributable"),
    ("Precisions", "precisions"),
    ("Golden", "golden"),
    ("Hosting", "hosting"),
    ("Gaps", "gaps"),
)


def _list(args: argparse.Namespace) -> int:
    records = _manifest(args).records
    if args.json:
        print(json.dumps([inventory_row(r) for r in records], indent=2))
    elif args.markdown:
        print("| " + " | ".join(title for title, _ in _COLUMNS) + " |")
        print("|" + "---|" * len(_COLUMNS))
        for row in map(inventory_row, records):
            row["source"] = _source(row["source"])
            print("| " + " | ".join(_cell(row[key]) for _, key in _COLUMNS) + " |")
    else:
        for record in records:
            print(f"{record.id}\t{','.join(record.precisions)}\t{record.task}\t{record.visibility}\t{record.title}")
    return 0


def _show(args: argparse.Namespace) -> int:
    try:
        record = _manifest(args).get(args.id)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 1
    print(json.dumps(dataclasses.asdict(record), indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .validate import ValidationError, validate

    if args.replay and args.no_signatures:
        print("--replay needs the signature checks; drop --no-signatures", file=sys.stderr)
        return 2
    root = Path(args.root)
    v1 = None if args.no_v1 else (args.v1 or root / "corpus-manifest-v1.json")
    if v1 is not None and not Path(v1).is_file():
        print(f"v1 manifest not found: {v1} (pass --no-v1 to skip the v1 check)", file=sys.stderr)
        return 2
    try:
        records = load_manifest(args.models or root / "models")
        validate(
            root, records, signatures=not args.no_signatures, replay=args.replay, v1=v1, complete=args.models is None
        )
    except (ImportError, ManifestError) as error:
        print(error, file=sys.stderr)
        return 2
    except ValidationError as error:
        for problem in error.problems:
            print(problem, file=sys.stderr)
        print(f"{len(error.problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"validated the records in {args.models or root / 'models'} against {root}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    from .hydrate import FetchError

    options = {"root": args.root} if args.root else {}
    try:
        record = _manifest(args).get(args.id)
        print(record.fetch(args.precision, **options))
        if args.golden:
            from .hydrate import fetch_file

            golden = record.precision(args.precision).golden
            if golden is None:
                raise KeyError(f"{record.id} has no golden for that precision")
            print(fetch_file(golden.file, **options))
        if args.card:
            print(record.card(**options))
    except (KeyError, FetchError) as error:
        print(error.args[0] if isinstance(error, KeyError) else error, file=sys.stderr)
        return 1
    return 0


def _guard(args: argparse.Namespace) -> int:
    import subprocess

    from .guard import scan
    from .hydrate import FetchError
    from .overlay import load_overlay

    try:
        hits = scan(
            Path(args.root), load_overlay(args.overlay), location=args.overlay, base=args.base, texts=tuple(args.text)
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr
        print(f"git failed: {' '.join(map(str, error.cmd[3:]))}: {(detail or '').strip()}", file=sys.stderr)
        return 2
    except (ManifestError, FetchError, OSError) as error:
        print(f"guard could not run: {error}", file=sys.stderr)
        return 2
    for hit in hits:
        print(f"{hit.where}: {hit.kind}", file=sys.stderr)
    if hits:
        print(f"{len(hits)} private name(s) found; do not publish", file=sys.stderr)
        return 1
    print("no private names found")
    return 0


def _steps(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t) if text else ()


def _pairs(values: list[str]) -> list[tuple[int, int]] | None:
    """``IN:OUT`` index pairs.

    Raises:
        ValueError: If a value is not two integers separated by ':'.
    """
    if not values:
        return None
    pairs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2 or not all(re.fullmatch("[0-9]+", p) for p in parts):
            raise ValueError(f"--pair expects IN:OUT input and output indices, got {value!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _golden_generate(args: argparse.Namespace) -> int:
    import numpy as np

    from . import golden
    from .manifest import IO, FileRef, Golden, Precision
    from .runtime import model_tensors, runtime_version, signature_names

    if args.print_record and args.data and not (args.source_uri and args.source_sha256):
        print("--print-record with --data needs --source-uri and --source-sha256", file=sys.stderr)
        return 2
    if args.record and args.pair:
        print("--pair cannot be combined with --record, which lists the state pairs", file=sys.stderr)
        return 2
    try:
        if args.record:
            record = _manifest(args).get(args.record)
            precision = record.precision(args.precision)
            io = record.io
        else:
            inputs, outputs = model_tensors(args.model)
            precision = Precision("model", FileRef("repo://model"), inputs, outputs, None)
            if args.pair:
                pairs = golden.pairs_by_index(inputs, outputs, _pairs(args.pair))
            else:
                pairs = golden.state_pairs_from_names(inputs, outputs, signature_names(args.model))
            io = IO("explicit_state" if pairs else "stateless", pairs)
        data = None
        if args.data:
            with np.load(args.data, allow_pickle=False) as loaded:
                bad = [key for key in loaded.files if not re.fullmatch(r"input_(0|[1-9][0-9]*)", key)]
                if bad:
                    raise ValueError(f"--data keys must be input_N, got {bad}")
                data = {int(key.removeprefix("input_")): loaded[key] for key in loaded.files}
            stateful = {i for i, _ in precision.pair_indices(io)}
            missing = [i for i in range(len(precision.inputs)) if i not in stateful and i not in data]
            if args.print_record and missing:
                raise ValueError(f"--data must give every data input when its source is recorded; missing {missing}")
        arrays = golden.generate(
            args.model,
            precision,
            io,
            kind=args.kind,
            steps=args.steps,
            resets=_steps(args.resets),
            data=data,
            seed=args.seed,
            resolver=args.resolver,
        )
    except (KeyError, ValueError, OSError) as error:
        print(error.args[0] if isinstance(error, KeyError) else error, file=sys.stderr)
        return 1
    golden.write(args.out, arrays)
    if args.print_record:
        source = {"uri": args.source_uri, "sha256": args.source_sha256} if args.data else {"seed": args.seed}
        meta = Golden(
            FileRef(f"lfs://{args.path or args.out.name}"),
            args.kind,
            args.steps,
            _steps(args.resets),
            source,
            "ai-edge-litert",
            runtime_version("ai-edge-litert") or "unknown",
            args.resolver,
        )
        print(json.dumps(golden.record_block(args.out, meta), indent=2))
    else:
        print(args.out)
    return 0


def _golden_check(args: argparse.Namespace) -> int:
    from . import golden

    try:
        pairs = _pairs(args.pair)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        problems = golden.check(
            args.model,
            args.golden,
            kind=args.kind,
            steps=args.steps,
            resets=_steps(args.resets),
            state_pairs=pairs,
            resolver=args.resolver,
            replay=not args.no_replay,
        )
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"golden matches {args.model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helia-zoo", description=__doc__)
    parser.add_argument("--models", type=Path, help="a models directory of records (default: the packaged records)")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list the records, or print the model inventory")
    formats = listing.add_mutually_exclusive_group()
    formats.add_argument("--json", action="store_true", help="the inventory as JSON")
    formats.add_argument("--markdown", action="store_true", help="the inventory as a Markdown table")
    listing.set_defaults(run=_list)
    show = commands.add_parser("show", help="print one record as JSON")
    show.add_argument("id", help="model ID")
    show.set_defaults(run=_show)
    check = commands.add_parser(
        "validate",
        help="validate the records against a hydrated checkout",
        description="Validate the records against a hydrated checkout. Unless the top-level --models option names "
        "other records, also refuse any tracked .tflite or .npz file that no record lists.",
    )
    check.add_argument(
        "--root",
        default=".",
        help="repository checkout with Git LFS hydrated, at the top of its git checkout (default: .)",
    )
    check.add_argument(
        "--replay", action="store_true", help="require each golden to replay exactly; needs the recorded LiteRT version"
    )
    check.add_argument("--no-signatures", action="store_true", help="skip checks that need LiteRT")
    check.add_argument("--v1", type=Path, help="v1 manifest to compare (default: <root>/corpus-manifest-v1.json)")
    check.add_argument("--no-v1", action="store_true", help="skip the v1 comparison")
    check.set_defaults(run=_validate)
    get = commands.add_parser("fetch", help="print the verified local path of a model (and its golden or card)")
    get.add_argument("id", help="model ID")
    get.add_argument("--precision", help="precision (default: the only one)")
    get.add_argument("--golden", action="store_true", help="also fetch the golden")
    get.add_argument("--card", action="store_true", help="also fetch the model card")
    get.add_argument("--root", help="hydrated checkout to read repository files from")
    get.set_defaults(run=_fetch)
    guard = commands.add_parser("guard", help="refuse if a checkout contains any name from a private overlay")
    guard.add_argument(
        "--overlay", required=True, help="overlay: a local directory or an hf://datasets/<org>/<repo>@<commit> root"
    )
    guard.add_argument("--root", default=".", help="checkout to scan (default: .)")
    guard.add_argument("--base", default="origin/main", help="scan commit messages in BASE..HEAD")
    guard.add_argument("--text", action="append", default=[], type=Path, help="extra text file, e.g. a PR body")
    guard.set_defaults(run=_guard)
    goldens = commands.add_parser("golden", help="generate or check golden fixtures").add_subparsers(
        dest="golden", required=True
    )
    make = goldens.add_parser("generate", help="run a model and write its golden NPZ")
    test = goldens.add_parser("check", help="check a golden NPZ against its model")
    for sub in (make, test):
        sub.add_argument("model", type=Path, help="TFLite model")
        sub.add_argument("--kind", default="single", choices=("single", "batch", "sequence"))
        sub.add_argument("--steps", type=int, default=1, help="calls in a batch or sequence")
        sub.add_argument("--resets", default="", help="comma-separated sequence steps that reset the state")
        sub.add_argument("--pair", action="append", default=[], help="state pair IN:OUT by I/O index (repeatable)")
        sub.add_argument("--resolver", default="builtin_ref", choices=("builtin_ref", "builtin"))
    make.add_argument("out", type=Path, help="NPZ to write")
    make.add_argument("--record", help="take tensors and state pairs from this model's record (ID)")
    make.add_argument("--precision", help="the record's precision")
    make.add_argument("--seed", type=int, default=42, help="seed for inputs not given by --data")
    make.add_argument(
        "--data",
        type=Path,
        help="NPZ of input_i arrays for data inputs: [steps, *shape], or shape for a single golden",
    )
    make.add_argument("--print-record", action="store_true", help="print the record's golden object")
    make.add_argument("--path", help="the golden's path in its model directory, for --print-record")
    make.add_argument("--source-uri", help="where --data came from, for --print-record")
    make.add_argument("--source-sha256", help="sha256 of the --source-uri file")
    make.set_defaults(run=_golden_generate)
    test.add_argument("golden", type=Path, help="NPZ to check")
    test.add_argument("--no-replay", action="store_true", help="check the arrays only")
    test.set_defaults(run=_golden_check)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except ManifestError as error:
        print(f"invalid record: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
