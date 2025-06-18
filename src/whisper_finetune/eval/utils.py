import re
import string
from typing import Dict, Set

WHITESPACE_REGEX = re.compile(r"[ \t]+")
NUMBER_REGEX = re.compile(r"^[0-9',.]+$")
NUMBER_DASH_REGEX = re.compile("[0-9]+[-\u2013\xad]")
DASH_NUMBER_REGEX = re.compile("[-\u2013\xad][0-9]+")

_CHAR_VOCAB_V0 = set(string.ascii_lowercase + string.digits + "äöü ")
_CHAR_VOCAB_V1 = set(string.ascii_lowercase + string.ascii_uppercase + string.digits + "äöüÄÖÜ" + " .,:")
_CHAR_VOCAB_V2 = set(string.ascii_lowercase + string.digits + "äöü" + " .,:")
_CHAR_VOCAB_V3 = set(string.ascii_lowercase + string.ascii_uppercase + string.digits + "äöüÄÖÜ" + " .,:-?!;")

_CHAR_LOOKUP_V0 = {
    "á": "a",
    "à": "a",
    "â": "a",
    "ç": "c",
    "é": "e",
    "è": "e",
    "ê": "e",
    "í": "i",
    "ì": "i",
    "î": "i",
    "ñ": "n",
    "ó": "o",
    "ò": "o",
    "ô": "o",
    "ú": "u",
    "ù": "u",
    "û": "u",
    "ș": "s",
    "ş": "s",
    "ß": "ss",
    "-": " ",
    # Not used consistently, better to replace with space as well:
    "–": " ",
    "/": " ",
}
_CHAR_LOOKUP_V1 = {
    **_CHAR_LOOKUP_V0,
    **{k.upper(): v.upper() for k, v in _CHAR_LOOKUP_V0.items()},
}
_CHAR_LOOKUP_V2 = _CHAR_LOOKUP_V1

_CHAR_LOOKUP_V3 = {
    "ß": "ss",
    "ç": "c",
    "á": "a",
    "à": "a",
    "â": "a",
    "é": "e",
    "è": "e",
    "ê": "e",
    "í": "i",
    "ì": "i",
    "î": "i",
    "ó": "o",
    "ò": "o",
    "ô": "o",
    "ú": "u",
    "ù": "u",
    "û": "u",
    "ñ": "n",
    "ș": "s",
    "\u2013": "-",
    "\xad": "-",
}

VOCAB_SPECS = {
    "v0": {
        "char_vocab": _CHAR_VOCAB_V0,
        "char_lookup": _CHAR_LOOKUP_V0,
        "transform_lowercase": True,
    },
    "v1": {
        "char_vocab": _CHAR_VOCAB_V1,
        "char_lookup": _CHAR_LOOKUP_V1,
        "transform_lowercase": False,
    },
    "v2": {
        "char_vocab": _CHAR_VOCAB_V2,
        "char_lookup": _CHAR_LOOKUP_V2,
        "transform_lowercase": False,
    },
    "v3": {
        "char_vocab": _CHAR_VOCAB_V3,
        "char_lookup": _CHAR_LOOKUP_V3,
        "transform_lowercase": False,
    },
}


def normalize_text(
    text: str,
    char_vocab: Set[str],
    char_lookup: Dict[str, str],
    transform_lowercase: bool = True,
) -> str:
    if transform_lowercase:
        text = text.lower()

    for q, r in char_lookup.items():
        text = text.replace(q, r)

    text = WHITESPACE_REGEX.sub(" ", text)
    text = "".join([char for char in text if char in char_vocab])
    text = WHITESPACE_REGEX.sub(" ", text)
    text = text.strip()
    return text
