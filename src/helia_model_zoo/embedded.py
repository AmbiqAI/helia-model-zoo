# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Extract literal TFLite bytes from a named C byte array; no inference or export."""

from __future__ import annotations

import re

_INTEGER = rb"(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)"
_LEXEME = re.compile(
    rb"/\*.*?\*/|//[^\r\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?P<unfinished>/\*|[\"'])",
    re.DOTALL,
)
_CONDITIONAL = re.compile(rb"(?m)^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif|else|endif)\b")
_GUARD = re.compile(
    rb"\s*#[ \t]*ifndef[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*\r?\n"
    rb"[ \t]*#[ \t]*define[ \t]+(?P=name)(?:[ \t]+1)?[ \t]*\r?\n"
    rb"(?P<body>.*)^[ \t]*#[ \t]*endif[ \t]*\s*",
    re.DOTALL | re.MULTILINE,
)


def _mask(match: re.Match[bytes]) -> bytes:
    if match.group("unfinished") is not None:
        raise ValueError("unterminated C comment or string/character literal")
    return re.sub(rb"[^\r\n]", b" ", match.group())


def _integer(value: bytes) -> int:
    return int(value, 16 if value.lower().startswith(b"0x") else 8 if value.startswith(b"0") else 10)


def extract_c_array(header: bytes, symbol: str) -> bytes:
    """Read one ``unsigned char``/``uint8_t`` array of literal bytes.

    Verify the header's provenance and the returned model's hash/size separately.
    Macros, expressions and conditional definitions are refused rather than
    evaluated. A canonical outer include guard is allowed.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError("expected a C array identifier")
    # C removes escaped newlines before recognizing comments and literals.
    text = _LEXEME.sub(_mask, re.sub(rb"\\\r?\n", b"", header))
    if guard := _GUARD.fullmatch(text):
        text = guard["body"]
    if _CONDITIONAL.search(text):
        raise ValueError("conditional array definitions need preprocessing; supply an unconditional literal header")
    text = re.sub(rb"(?m)^[ \t]*#[^\r\n]*", b"", text)
    pattern = (
        rb"\b(?:unsigned\s+char|uint8_t)\s+"
        + re.escape(symbol.encode("ascii"))
        + rb"\s*\[\s*("
        + _INTEGER
        + rb")?\s*\]\s*(?:__attribute__\s*\(\([^;={}]*\)\)\s*)?=\s*\{([^{}]*)\}\s*;"
    )
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one literal byte array named {symbol}, found {len(matches)}")
    extent, body = matches[0]
    tokens = body.strip().removesuffix(b",").split(b",")
    values = []
    for token in tokens:
        token = token.strip()
        if not re.fullmatch(_INTEGER, token):
            raise ValueError(f"{symbol}: initializer must contain literal bytes only")
        value = _integer(token)
        if value > 255:
            raise ValueError(f"{symbol}: initializer contains a value outside 0..255")
        values.append(value)
    model = bytes(values)
    if extent and _integer(extent) != len(model):
        raise ValueError(f"{symbol}: declared extent differs from literal byte count")
    if model[4:8] != b"TFL3":
        raise ValueError(f"{symbol}: extracted bytes have no TFLite file identifier")
    return model
