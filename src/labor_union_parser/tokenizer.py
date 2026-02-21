"""Tokenizer for structured classifier input."""

from .char_cnn import tokenize_to_chars
from .classifier import MAX_CHARS_PER_TOKEN, MAX_TOKENS


def smart_truncate_nonspace(text, max_tokens=MAX_TOKENS):
    """Tokenize, drop spaces, keep first N tokens, recover lost numbers.

    Returns list of dicts with keys: chars, token, is_num, token_type
    """
    full_chars, full_tokens, full_is_num, full_token_types, _ = tokenize_to_chars(
        text, 999
    )

    nonspace = []
    for i, tt in enumerate(full_token_types):
        if full_tokens[i] and tt != 2:  # not space, not empty
            nonspace.append(
                {
                    "chars": full_chars[i],
                    "token": full_tokens[i],
                    "is_num": full_is_num[i],
                    "token_type": tt,
                }
            )

    trunc = nonspace[:max_tokens]

    # Recover lost numbers
    trunc_numbers = {d["token"] for d in trunc if d["is_num"]}
    lost_numbers = [
        d for d in nonspace if d["is_num"] and d["token"] not in trunc_numbers
    ]

    if lost_numbers:
        replace_positions = []
        for i in range(len(trunc) - 1, -1, -1):
            if not trunc[i]["is_num"] and trunc[i]["token"]:
                replace_positions.append(i)
                if len(replace_positions) >= len(lost_numbers):
                    break
        replace_positions.reverse()
        for pos, lost in zip(replace_positions, lost_numbers):
            trunc[pos] = lost

    # Pad
    while len(trunc) < max_tokens:
        trunc.append(
            {
                "chars": [0] * MAX_CHARS_PER_TOKEN,
                "token": "",
                "is_num": 0,
                "token_type": 4,
            }
        )

    return trunc
