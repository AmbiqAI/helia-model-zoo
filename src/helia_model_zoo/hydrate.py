# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Fetch manifest files into a sha256-keyed cache, verifying every byte.

Standard library only, except ``hf://`` sources, which need the ``hf`` extra (huggingface_hub).
Credentials are never read, printed or stored here: Hugging Face access uses huggingface_hub's
own token resolution (``HF_TOKEN`` or ``hf auth login``).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import tempfile
import urllib.request
from functools import cache
from importlib import metadata
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from .manifest import FileRef, parse_hf

REPOSITORY = "AmbiqAI/helia-model-zoo"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_CHUNK = 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}")


class FetchError(RuntimeError):
    """A file could not be fetched, or its bytes did not match the manifest."""


@cache
def _install() -> dict:
    """This package's PEP 610 ``direct_url.json``, or an empty dict."""
    try:
        text = metadata.distribution("helia-model-zoo").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return {}
    return json.loads(text) if text else {}


def installed_revision() -> str | None:
    """The repository commit whose files this manifest describes.

    ``HELIA_ZOO_REVISION`` if set, else the commit this package was installed from by git SHA.
    """
    return os.environ.get("HELIA_ZOO_REVISION") or _install().get("vcs_info", {}).get("commit_id")


def local_checkout() -> Path | None:
    """A local checkout to read repository files from.

    ``HELIA_ZOO_ROOT`` if set, else the checkout of an editable install.
    """
    if os.environ.get("HELIA_ZOO_ROOT"):
        return Path(os.environ["HELIA_ZOO_ROOT"])
    install = _install()
    if install.get("dir_info", {}).get("editable") and install.get("url", "").startswith("file://"):
        return Path(unquote(urlparse(install["url"]).path))
    return None


def cache_dir() -> Path:
    """``HELIA_ZOO_CACHE``, or ``~/.cache/helia-model-zoo``."""
    return Path(os.environ.get("HELIA_ZOO_CACHE") or Path.home() / ".cache" / "helia-model-zoo")


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _is_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def _matches(path: Path, ref: FileRef) -> bool:
    if ref.bytes is not None and path.stat().st_size != ref.bytes:
        return False
    return ref.sha256 is None or sha256_file(path) == ref.sha256


def _url(ref: FileRef, commit: str) -> str:
    path = quote(ref.path)
    if ref.scheme == "lfs":
        return f"https://media.githubusercontent.com/media/{REPOSITORY}/{commit}/{path}"
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{path}"


def _download_url(url: str, destination: Path, limit: int | None = None) -> None:
    """Write the body of an HTTPS GET to ``destination``; no credentials are sent.

    Stops as soon as the body exceeds ``limit`` bytes.
    """
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as out:
        written = 0
        while chunk := response.read(_CHUNK):
            written += len(chunk)
            if limit is not None and written > limit:
                raise FetchError(f"{url}: more than the expected {limit} bytes")
            out.write(chunk)


def _download_hf(uri: str, destination: Path, anonymous: bool) -> None:
    """Write a Hugging Face file to ``destination`` with huggingface_hub's own authentication."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FetchError("hf:// sources need huggingface_hub: install helia-model-zoo[hf]") from error
    location = parse_hf(uri)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        try:
            downloaded = hf_hub_download(
                location.repo_id,
                location.path,
                repo_type=location.repo_type,
                revision=location.revision,
                local_dir=scratch,
                token=False if anonymous else None,
            )
        except Exception as error:  # huggingface_hub raises many types; report which file failed
            raise FetchError(f"could not download {uri}: {type(error).__name__}") from error
        os.replace(downloaded, destination)


def _download(ref: FileRef, destination: Path, commit: str | None, anonymous: bool) -> None:
    if ref.scheme == "hf":
        _download_hf(ref.uri, destination, anonymous)
        return
    url = ref.uri if ref.scheme == "https" else _url(ref, commit)
    try:
        _download_url(url, destination, ref.bytes)
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise FetchError(f"could not download {url}: {type(error).__name__}: {error}") from error


def _target(ref: FileRef, cache: Path, commit: str | None) -> Path:
    name = Path(ref.path).name
    if ref.sha256 is not None:
        return cache / ref.sha256 / name
    # Unhashed text: keyed by the revision that pins it.
    pinned = parse_hf(ref.uri).revision if ref.scheme == "hf" else commit
    return cache / "text" / ref.scheme / pinned / ref.path


def fetch_file(
    ref: FileRef,
    *,
    root: Path | None = None,
    revision: str | None = None,
    cache: Path | None = None,
    anonymous: bool = False,
) -> Path:
    """Return a local path whose bytes match ``ref``, downloading into the cache if needed.

    A repository file comes from ``root`` (default :func:`local_checkout`) when hydrated there,
    else from GitHub at ``revision`` (default :func:`installed_revision`). A cached copy is re-verified on every
    call and replaced if it no longer matches. A download is verified before it is moved into
    place, so the cache never holds an unverified file.

    Args:
        ref: The manifest file to fetch.
        root: A checkout of this repository.
        revision: The repository commit to download repository files from.
        cache: The cache directory (default :func:`cache_dir`).
        anonymous: Download without credentials, as public CI does.

    Raises:
        FetchError: If the file cannot be found or downloaded, or its bytes do not match.
    """
    if ref.in_repository:
        root = local_checkout() if root is None else root
        if root is not None:
            try:
                local = ref.resolve(root)
            except ValueError as error:
                raise FetchError(str(error)) from None
            if local.is_file() and not _is_pointer(local):
                if not _matches(local, ref):
                    raise FetchError(f"{local} does not match the manifest's size and sha256 for {ref.uri}")
                return local
        revision = revision or installed_revision()
        if revision is None:
            raise FetchError(
                f"cannot locate {ref.uri}: no hydrated checkout and no revision; install the package from a "
                "git SHA, or set HELIA_ZOO_ROOT or HELIA_ZOO_REVISION"
            )
        if not _REVISION.fullmatch(revision):
            raise FetchError(f"revision must be a full 40-hex commit, got {revision!r}")
    cache = cache_dir() if cache is None else Path(cache)
    target = _target(ref, cache, revision)
    if target.is_file() and _matches(target, ref):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    os.close(handle)
    partial = Path(name)
    try:
        _download(ref, partial, revision, anonymous)
        if ref.bytes is not None and partial.stat().st_size != ref.bytes:
            raise FetchError(f"{ref.uri}: downloaded {partial.stat().st_size} bytes, expected {ref.bytes}")
        if ref.sha256 is not None and (actual := sha256_file(partial)) != ref.sha256:
            raise FetchError(f"{ref.uri}: sha256 mismatch: expected {ref.sha256}, got {actual}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target
