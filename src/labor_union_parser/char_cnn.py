"""Character-level tokenization for text preprocessing.

Tokenizes text into character ID sequences with token type classification
(word, number, space, punctuation, padding).
"""

MAX_TOKENS = 70
MAX_CHARS_PER_TOKEN = 20

# Character vocabulary: lowercase letters, digits, common punctuation
CHAR_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
}
# Add lowercase letters
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    CHAR_VOCAB[c] = len(CHAR_VOCAB)
# Add digits
for i, c in enumerate("0123456789"):
    CHAR_VOCAB[c] = len(CHAR_VOCAB)
# Add common punctuation and space
for c in " -/&,.()'\"#:":
    CHAR_VOCAB[c] = len(CHAR_VOCAB)


def chars_to_ids(token: str, max_chars: int = MAX_CHARS_PER_TOKEN) -> list[int]:
    """Convert a token to character IDs."""
    token = token.lower()
    ids = []
    for c in token[:max_chars]:
        ids.append(CHAR_VOCAB.get(c, CHAR_VOCAB["<UNK>"]))
    # Pad to max_chars
    ids = ids + [0] * (max_chars - len(ids))
    return ids


def tokenize_to_chars(
    text: str, max_tokens: int = MAX_TOKENS, max_chars: int = MAX_CHARS_PER_TOKEN
) -> tuple[list[list[int]], list[str], list[int], list[int]]:
    """Tokenize text and convert to character IDs.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        max_chars: Maximum characters per token

    Returns:
        char_ids: [max_tokens, max_chars] character IDs
        tokens: List of token strings
        is_number: [max_tokens] 1 if token is a number, 0 otherwise
        token_type: [max_tokens] 0=word, 1=number, 2=space, 3=punct, 4=pad
    """
    import re

    # Pre-process: merge space-separated single letters (A F G E → AFGE)
    # Pattern matches single letters separated by spaces, not adjacent to other letters
    def merge_spaced_letters(m):
        return "".join(m.group(0).split())

    text = re.sub(
        r"(?<![A-Za-z])([A-Za-z])(?:\s+([A-Za-z]))+(?![A-Za-z])",
        merge_spaced_letters,
        text,
    )

    # Regex pattern:
    # 1. Acronyms: single letters separated by periods (I.B.E.W.)
    # 2. Words: consecutive letters
    # 3. Numbers: digits
    # 4. Spaces: whitespace (normalized to single space)
    # 5. Period followed by non-space (keeps No.123, drops No. 123)
    # 6. Other punctuation (not period)
    pattern = (
        r"([A-Za-z](?:\.[A-Za-z])+\.?)|([a-zA-Z]+)|(\d+)|(\s+)|(\.(?=\S))|([^\s\w.])"
    )

    tokens = []
    is_number = []
    token_type = []  # 0=word, 1=number, 2=space, 3=punct

    for match in re.finditer(pattern, text.lower()):
        if match.group(1):  # acronym - strip periods
            tokens.append(match.group(1).replace(".", ""))
            is_number.append(0)
            token_type.append(0)
        elif match.group(2):  # word
            tokens.append(match.group(2))
            is_number.append(0)
            token_type.append(0)
        elif match.group(3):  # number
            num = match.group(3).lstrip("0") or "0"
            tokens.append(num)
            is_number.append(1)
            token_type.append(1)
        elif match.group(4):  # space - normalize to single space
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

    # Truncate to max_tokens
    tokens = tokens[:max_tokens]
    is_number = is_number[:max_tokens]
    token_type = token_type[:max_tokens]

    # Convert to character IDs
    char_ids = []
    for token in tokens:
        char_ids.append(chars_to_ids(token, max_chars))

    # Pad to max_tokens
    while len(char_ids) < max_tokens:
        char_ids.append([0] * max_chars)
        tokens.append("")  # empty token for padding
        is_number.append(0)
        token_type.append(4)  # pad

    return char_ids, tokens, is_number, token_type
