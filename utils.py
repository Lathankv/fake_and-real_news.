import re


def clean(text: str) -> str:
    """Normalize article text the same way for training and prediction."""
    text = str(text)
    # The Kaggle dataset's real articles start with "CITY (Reuters) -".
    # Removing it stops the model from just learning that one giveaway.
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()
