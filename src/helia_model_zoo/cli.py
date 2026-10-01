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
    from .overlay import load_overlay

    try:
        hits = scan(
            Path(args.root), load_overlay(args.overlay), location=args.overlay, base=args.base, texts=tuple(args.text)
        )
    except subprocess.CalledProcessError as error:
        print(f"git failed: {' '.join(error.cmd[3:])}: {error.stderr.strip()}", file=sys.stderr)
        return 2
    for hit in hits:
        print(f"{hit.where}: {hit.kind}", file=sys.stderr)
    if hits:
        print(f"{len(hits)} private name(s) found; do not publish", file=sys.stderr)
        return 1
    print("no private names found")
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
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except ManifestError as error:
        print(f"invalid manifest: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
