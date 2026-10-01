# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""The ``helia-zoo`` command line."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .manifest import ManifestError, load_manifest


def _manifest(args: argparse.Namespace):
    """``--manifest`` if given, else the packaged manifest plus any ``HELIA_ZOO_OVERLAY``."""
    if args.manifest is not None:
        return load_manifest(args.manifest)
    from . import manifest

    return manifest()


def _list(args: argparse.Namespace) -> int:
    for entry in _manifest(args).entries:
        print(f"{entry.id}\t{','.join(entry.precisions)}\t{entry.tier}\t{entry.visibility}\t{entry.title}")
    return 0


def _show(args: argparse.Namespace) -> int:
    try:
        entry = _manifest(args).get(args.id)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 1
    print(json.dumps(dataclasses.asdict(entry), indent=2))
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
        validate(root, load_manifest(args.manifest), signatures=not args.no_signatures, replay=args.replay, v1=v1)
    except ImportError as error:
        print(error, file=sys.stderr)
        return 2
    except ValidationError as error:
        for problem in error.problems:
            print(problem, file=sys.stderr)
        print(f"{len(error.problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"validated {args.manifest or 'the packaged manifest'} against {root}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    from .hydrate import FetchError

    options = {"root": args.root} if args.root else {}
    try:
        entry, aliased = _manifest(args).resolve(args.id)
        precision = args.precision or (aliased.name if aliased else None)
        print(entry.fetch(precision, **options))
        if args.golden:
            from .hydrate import fetch_file

            golden = entry.precision(precision).golden
            if golden is None:
                raise KeyError(f"{entry.id} has no golden for that precision")
            print(fetch_file(golden.file, **options))
        if args.card:
            print(entry.card(**options))
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
    if not values:
        return None
    return [tuple(int(x) for x in value.split(":", 1)) for value in values]


def _golden_generate(args: argparse.Namespace) -> int:
    import numpy as np

    from . import golden
    from .manifest import IO, FileRef, Golden, Precision, StatePair
    from .runtime import model_tensors, runtime_version, signature_names

    try:
        if args.entry:
            entry, aliased = _manifest(args).resolve(args.entry)
            precision = entry.precision(args.precision or (aliased.name if aliased else None))
            io = entry.io
        else:
            inputs, outputs = model_tensors(args.model)
            precision = Precision("model", FileRef("repo://model"), inputs, outputs, None)
            if args.pair:
                pairs = tuple(
                    StatePair(i, o, "zeros", inputs[i].scale == outputs[o].scale) for i, o in _pairs(args.pair)
                )
            else:
                pairs = golden.state_pairs_from_names(inputs, outputs, signature_names(args.model))
            io = IO("explicit_state" if pairs else "stateless", pairs)
        data = None
        if args.data:
            with np.load(args.data, allow_pickle=False) as loaded:
                data = {int(key.removeprefix("input_")): loaded[key] for key in loaded.files}
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
    except (KeyError, golden.GoldenError) as error:
        print(error.args[0] if isinstance(error, KeyError) else error, file=sys.stderr)
        return 1
    golden.write(args.out, arrays)
    if args.print_manifest:
        if args.data and not (args.source_uri and args.source_sha256):
            print("--print-manifest with --data needs --source-uri and --source-sha256", file=sys.stderr)
            return 2
        source = {"uri": args.source_uri, "sha256": args.source_sha256} if args.data else {"seed": args.seed}
        meta = Golden(
            FileRef(args.uri or f"lfs://{args.out.name}"),
            args.kind,
            args.steps,
            _steps(args.resets),
            source,
            "ai-edge-litert",
            runtime_version("ai-edge-litert") or "unknown",
            args.resolver,
        )
        print(json.dumps(golden.manifest_block(args.out, meta), indent=2))
    else:
        print(args.out)
    return 0


def _golden_check(args: argparse.Namespace) -> int:
    from . import golden

    problems = golden.check(
        args.model,
        args.golden,
        kind=args.kind,
        steps=args.steps,
        resets=_steps(args.resets),
        state_pairs=_pairs(args.pair),
        resolver=args.resolver,
        replay=not args.no_replay,
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"golden matches {args.model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helia-zoo", description=__doc__)
    parser.add_argument("--manifest", type=Path, help="a manifest@2 file (default: the packaged manifest)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list entries").set_defaults(run=_list)
    show = commands.add_parser("show", help="print one entry as JSON")
    show.add_argument("id", help="model ID or v1 alias")
    show.set_defaults(run=_show)
    check = commands.add_parser("validate", help="validate the manifest against a hydrated checkout")
    check.add_argument("--root", default=".", help="repository checkout with Git LFS hydrated (default: .)")
    check.add_argument(
        "--replay", action="store_true", help="require each golden to replay exactly; needs the recorded LiteRT version"
    )
    check.add_argument("--no-signatures", action="store_true", help="skip checks that need LiteRT")
    check.add_argument("--v1", type=Path, help="v1 manifest to compare (default: <root>/corpus-manifest-v1.json)")
    check.add_argument("--no-v1", action="store_true", help="skip the v1 comparison")
    check.set_defaults(run=_validate)
    get = commands.add_parser("fetch", help="print the verified local path of a model (and its golden or card)")
    get.add_argument("id", help="model ID or alias")
    get.add_argument("--precision", help="precision (default: the alias's, or the only one)")
    get.add_argument("--golden", action="store_true", help="also fetch the golden")
    get.add_argument("--card", action="store_true", help="also fetch the model card")
    get.add_argument("--root", help="hydrated checkout to read repository files from")
    get.set_defaults(run=_fetch)
    guard = commands.add_parser("guard", help="refuse if a checkout contains any name from a private overlay")
    guard.add_argument("--overlay", required=True, help="overlay manifest: a local path or an hf:// URI")
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
    make.add_argument("--entry", help="take tensors and state pairs from this manifest entry (ID or alias)")
    make.add_argument("--precision", help="the entry's precision")
    make.add_argument("--seed", type=int, default=42, help="seed for inputs not given by --data")
    make.add_argument("--data", type=Path, help="NPZ of input_i arrays [steps, *shape] for data inputs")
    make.add_argument("--print-manifest", action="store_true", help="print the manifest golden block")
    make.add_argument("--uri", help="the golden's manifest URI, for --print-manifest")
    make.add_argument("--source-uri", help="where --data came from, for --print-manifest")
    make.add_argument("--source-sha256", help="sha256 of the --source-uri file")
    make.set_defaults(run=_golden_generate)
    test.add_argument("golden", type=Path, help="NPZ to check")
    test.add_argument("--no-replay", action="store_true", help="check the arrays only")
    test.set_defaults(run=_golden_check)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except ManifestError as error:
        print(f"invalid manifest: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
