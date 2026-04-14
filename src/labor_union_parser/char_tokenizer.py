"""Text tokenization with token type classification.

Splits text into tokens classified as word, number, space, punctuation,
or padding. Handles acronyms (I.B.E.W.), spaced single letters (A F G E),
and number normalization (strip leading zeros).
"""

import re

MAX_TOKENS = 70


def _merge_spaced_letters(m):
    return "".join(m.group(0).split())


def tokenize(text, max_tokens=MAX_TOKENS):
    """Tokenize text into classified tokens.

    Returns:
        tokens: list of str (padded to max_tokens with empty strings)
        is_number: list of int (1 if number, 0 otherwise)
        token_type: list of int (0=word, 1=number, 2=space, 3=punct, 4=pad)
    """
    # Pre-process: merge space-separated single letters (A F G E → AFGE)
    text = re.sub(
        r"(?<![A-Za-z])([A-Za-z])(?:\s+([A-Za-z]))+(?![A-Za-z])",
        _merge_spaced_letters,
        text,
    )

    # Regex: acronyms, words, numbers, spaces, attached periods, punctuation
    pattern = (
        r"([A-Za-z](?:\.[A-Za-z])+\.?)|([a-zA-Z]+)|(\d+)|(\s+)|(\.(?=\S))|([^\s\w.])"
    )

    tokens = []
    is_number = []
    token_type = []

    for match in re.finditer(pattern, text.lower()):
        if match.group(1):  # acronym - strip periods
            tokens.append(match.group(1).replace(".", ""))
            is_number.append(0)
            token_type.append(0)
        elif match.group(2):  # word
            tokens.append(match.group(2))
            is_number.append(0)
            token_type.append(0)
        elif match.group(3):  # number - strip leading zeros
            tokens.append(match.group(3).lstrip("0") or "0")
            is_number.append(1)
            token_type.append(1)
        elif match.group(4):  # space
            tokens.append(" ")
            is_number.append(0)
            token_type.append(2)
        elif match.group(5):  # period (followed by non-space)
            tokens.append(".")
            is_number.append(0)
            token_type.append(3)
        elif match.group(6):  # other punct
            tokens.append(match.group(6))
            is_number.append(0)
            token_type.append(3)

    # Truncate
    tokens = tokens[:max_tokens]
    is_number = is_number[:max_tokens]
    token_type = token_type[:max_tokens]

    # Pad
    while len(tokens) < max_tokens:
        tokens.append("")
        is_number.append(0)
        token_type.append(4)

    return tokens, is_number, token_type
