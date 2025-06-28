import re

LIGATURES = {
    "ɻ": "fl",
    "ɹ": "fi",
    "ɽ": "ffi",
    "ʀ": "ffl",
    "ʃ": "ff",
}


def replace_ligatures(text: str) -> str:
    """
    Replace common ligatures in the text with their expanded forms.
    Args:
        text (str): The input text containing ligatures.
    Returns:
        str: The text with ligatures replaced by their expanded forms.
    """
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def clean_text(text: str) -> str:
    """
    Clean the input text by removing unnecessary characters and formatting.
    Removes newlines, replaces multiple spaces with a single space,
    and replaces ligatures with their expanded forms.
    Args:
        text (str): The input text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)  # replace multiple spaces with a single space
    text = replace_ligatures(text)
    return text.strip()
