"""IMDb dataset loading with reproducible, leakage-safe splits."""
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Remove HTML and normalize whitespace; preserve negation and punctuation."""
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def load_imdb(seed=42, validation_fraction=0.1, limit=None, source="data/IMDB Dataset.csv"):
    """Return train/validation/test data with columns text and label (0/1).

    Kaggle CSV: remove duplicates before all splits; reserve 20% for final test,
    then split the remainder into train/validation. validation_fraction is the
    fraction of the training pool (default 0.1 => 72/8/20 overall).
    Hugging Face alternative 'stanfordnlp/imdb': preserves official 25k test.
    """
    source = str(source)
    if source == "stanfordnlp/imdb":
        from datasets import load_dataset
        ds = load_dataset(source)
        pool = ds["train"].to_pandas()[["text", "label"]]
        test = ds["test"].to_pandas()[["text", "label"]]
    else:
        csv_path = Path(source)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"IMDb CSV not found at {csv_path}. Put the downloaded Kaggle "
                "file at data/IMDB Dataset.csv or edit dataset in configs/default.yaml."
            )
        raw = pd.read_csv(csv_path, usecols=["review", "sentiment"])
        raw = raw.dropna(subset=["review", "sentiment"]).copy()
        raw["text"] = raw["review"].map(clean_text)
        raw["label"] = raw["sentiment"].str.lower().map({"negative": 0, "positive": 1})
        if raw["label"].isna().any():
            raise ValueError("CSV sentiment column must contain positive/negative labels.")
        raw = raw[["text", "label"]]
        # Do this before splitting so duplicated reviews cannot cross partitions.
        raw = raw.drop_duplicates(subset=["text"]).reset_index(drop=True)
        pool, test = train_test_split(
            raw, test_size=0.2, stratify=raw["label"], random_state=seed
        )
    if limit is not None:
        if int(limit) < 4:
            raise ValueError("--limit must be >= 4 and is DEBUG-ONLY.")
        n_train = min(len(pool), int(limit))
        n_test = min(len(test), max(2, int(limit)//2))
        pool, _ = train_test_split(pool, train_size=n_train,
                                   stratify=pool["label"], random_state=seed)
        test, _ = train_test_split(test, train_size=n_test,
                                   stratify=test["label"], random_state=seed)
    fit, val = train_test_split(
        pool, test_size=validation_fraction, stratify=pool["label"], random_state=seed
    )
    return [
        frame.reset_index(drop=True).assign(text=lambda x: x["text"].map(clean_text))
        for frame in (fit, val, test)
    ]


def save_split_manifest(train, val, test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        name: {
            "rows": len(frame),
            "labels": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        }
        for name, frame in [("train", train), ("validation", val), ("test", test)]
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(counts, indent=2))
    return counts
