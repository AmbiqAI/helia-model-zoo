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


def _list(args: argparse.Namespace) -> int:
    for entry in load_manifest(args.manifest).entries:
        print(f"{entry.id}\t{','.join(entry.precisions)}\t{entry.tier}\t{entry.visibility}\t{entry.title}")
    return 0


def _show(args: argparse.Namespace) -> int:
    try:
        entry = load_manifest(args.manifest).get(args.id)
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
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except ManifestError as error:
        print(f"invalid manifest: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
