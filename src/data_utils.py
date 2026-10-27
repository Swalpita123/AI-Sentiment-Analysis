"""Dataset loading and lightweight text preprocessing helpers."""
from datasets import load_dataset
import re


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving words and punctuation for the transformer."""
    text = str(text or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_imdb_dataset(train_samples: int = 0, eval_samples: int = 0, seed: int = 42):
    """
    Download the public IMDb sentiment dataset from Hugging Face Datasets.
    Set sample counts to 0 to use the full official split.
    """
    # dataset = load_dataset("imdb")
    dataset = load_dataset("stanfordnlp/imdb")
    train_data = dataset["train"].shuffle(seed=seed)
    test_data = dataset["test"].shuffle(seed=seed)

    if train_samples > 0:
        train_data = train_data.select(range(min(train_samples, len(train_data))))
    if eval_samples > 0:
        test_data = test_data.select(range(min(eval_samples, len(test_data))))

    def normalize(batch):
        batch["text"] = [clean_text(x) for x in batch["text"]]
        return batch

    train_data = train_data.map(normalize, batched=True)
    test_data = test_data.map(normalize, batched=True)
    return train_data, test_data
