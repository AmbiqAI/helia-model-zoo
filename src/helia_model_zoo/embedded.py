# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Extract literal TFLite bytes from a named C byte array; no inference or export."""

from __future__ import annotations

import re


def extract_c_array(header: bytes, symbol: str) -> bytes:
    """Read one ``unsigned char``/``uint8_t`` array of literal bytes.

    Verify the header's provenance and the returned model's hash/size separately.
    Macros, expressions and non-byte literals are refused rather than evaluated.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError("expected a C array identifier")
    text = re.sub(rb"/\*.*?\*/|//[^\r\n]*", b"", header, flags=re.DOTALL)
    pattern = (
        rb"\b(?:unsigned\s+char|uint8_t)\s+"
        + re.escape(symbol.encode("ascii"))
        + rb"\s*\[\s*([0-9]*)\s*\]\s*(?:__attribute__\s*\(\([^;={}]*\)\)\s*)?=\s*\{([^{}]*)\}\s*;"
    )
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one literal byte array named {symbol}, found {len(matches)}")
    extent, body = matches[0]
    tokens = body.strip().removesuffix(b",").split(b",")
    values = []
    for token in tokens:
        token = token.strip()
        if not re.fullmatch(rb"(?:0[xX][0-9a-fA-F]{1,2}|0|[1-9][0-9]{0,2})", token):
            raise ValueError(f"{symbol}: initializer must contain literal bytes only")
        value = int(token, 16 if token.lower().startswith(b"0x") else 10)
        if value > 255:
            raise ValueError(f"{symbol}: initializer contains a value outside 0..255")
        values.append(value)
    model = bytes(values)
    if extent and int(extent) != len(model):
        raise ValueError(f"{symbol}: declared extent differs from literal byte count")
    if model[4:8] != b"TFL3":
        raise ValueError(f"{symbol}: extracted bytes have no TFLite file identifier")
    return model
