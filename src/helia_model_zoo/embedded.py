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
_SPACE = rb"[ \t\v\f]"
_DIRECTIVE = rb"(?m)^" + _SPACE + rb"*#" + _SPACE + rb"*"
_TRIGRAPHS = dict(zip(b"=/'()!<>-", (b"#", b"\\", b"^", b"[", b"]", b"|", b"{", b"}", b"~"), strict=True))
_DIGRAPHS = {b"%:%:": b"##", b"<:": b"[", b":>": b"]", b"<%": b"{", b"%>": b"}", b"%:": b"#"}
_GUARD = re.compile(
    rb"\s*#[ \t\v\f]*ifndef[ \t\v\f]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t\v\f]*\n"
    rb"[ \t\v\f]*#[ \t\v\f]*define[ \t\v\f]+(?P=name)(?:[ \t\v\f]+1)?[ \t\v\f]*\n"
    rb"(?P<body>.*)^[ \t\v\f]*#[ \t\v\f]*endif[ \t\v\f]*\s*",
    re.DOTALL | re.MULTILINE,
)


def _mask(match: re.Match[bytes]) -> bytes:
    if match.group("unfinished") is not None:
        raise ValueError("unterminated C comment or string/character literal")
    return re.sub(rb"[^\r\n]", b" ", match.group())


def _integer(value: bytes) -> int:
    return int(value, 16 if value.lower().startswith(b"0x") else 8 if value.startswith(b"0") else 10)


def extract_c_array(header: bytes, symbol: str) -> bytes:
    """Read one global ``unsigned char``/``uint8_t`` array of literal bytes.

    Verify the header's provenance and the returned model's hash/size separately.
    Macros, expressions and conditional definitions are refused rather than
    evaluated. A canonical outer include guard is allowed.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError("expected a C array identifier")
    # C translation replaces trigraphs before splicing lines and recognizing literals.
    text = header.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    text = re.sub(rb"\?\?([=/\'()!<>-])", lambda match: _TRIGRAPHS[match[1][0]], text)
    text = _LEXEME.sub(_mask, re.sub(rb"\\\n", b"", text))
    text = re.sub(rb"%:%:|<:|:>|<%|%>|%:", lambda match: _DIGRAPHS[match[0]], text)
    if guard := _GUARD.fullmatch(text):
        text = guard["body"]
    if re.search(_DIRECTIVE, text):
        raise ValueError("preprocessing beyond a canonical outer include guard is unsupported")
    pattern = (
        rb"\b(?:unsigned\s+char|uint8_t)\s+"
        + re.escape(symbol.encode("ascii"))
        + rb"\s*\[\s*("
        + _INTEGER
        + rb")?\s*\]\s*(?:__attribute__\s*\(\([^;={}]*\)\)\s*)?=\s*\{([^{}]*)\}\s*;"
    )
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one literal byte array named {symbol}, found {len(matches)}")
    # A declaration in a macro argument or function/block is not a global array.
    depth = {ord("("): 0, ord("{"): 0}
    closes = {ord(")"): ord("("), ord("}"): ord("{")}
    for char in text[: matches[0].start()]:
        if char in depth:
            depth[char] += 1
        elif char in closes:
            depth[closes[char]] -= 1
            if depth[closes[char]] < 0:
                raise ValueError(f"{symbol}: malformed declaration context")
    if any(depth.values()):
        raise ValueError(f"{symbol}: expected a global array outside macro arguments or blocks")
    extent, body = matches[0].groups()
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
