"""IMDb loading and leakage-safe train/validation/test split."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Remove markup/normalize spaces; retain punctuation, case and negation cues."""
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb(seed=42, validation_fraction=0.1, limit=None):
    """Returns train, val, test DataFrames. Official test data are NEVER used for tuning."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].to_pandas()[["text", "label"]]
    test = ds["test"].to_pandas()[["text", "label"]]
    if limit:
        # sample class-balanced subsets for fast smoke runs; never use for final report
        n_train = max(2, min(len(train), int(limit)))
        n_test = max(2, min(len(test), int(limit) // 2))
        train, _ = train_test_split(train, train_size=n_train, stratify=train.label, random_state=seed)
        test, _ = train_test_split(test, train_size=n_test, stratify=test.label, random_state=seed)
    fit, val = train_test_split(train, test_size=validation_fraction, stratify=train.label, random_state=seed)
    return [d.reset_index(drop=True).assign(text=lambda x: x.text.map(clean_text)) for d in (fit, val, test)]


def save_split_manifest(train, val, test, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: {"rows": len(frame), "labels": {str(k): int(v) for k,v in frame.label.value_counts().items()}} for name,frame in [("train",train),("validation",val),("test",test)]}
    import json
    (output_dir / "dataset_manifest.json").write_text(json.dumps(counts, indent=2))
    return counts