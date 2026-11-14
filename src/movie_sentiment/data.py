"""IMDb dataset loading with reproducible, leakage-safe splits."""
import json
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def load_imdb(seed=42, validation_fraction=0.1, limit=None, source="data/IMDB Dataset.csv"):
    """Return train, validation, test with text + binary label.

    Kaggle CSV: drop duplicate reviews BEFORE splitting, stratified 72/8/20.
    HF source: retain official IMDb test partition.
    """
    source = str(source)
    if source == "stanfordnlp/imdb":
        from datasets import load_dataset
        ds = load_dataset(source)
        pool = ds["train"].to_pandas()[["text", "label"]]
        test = ds["test"].to_pandas()[["text", "label"]]
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"IMDb CSV not found at {path}; put Kaggle CSV at data/IMDB Dataset.csv")
        raw = pd.read_csv(path, usecols=["review", "sentiment"])
        raw = raw.dropna(subset=["review", "sentiment"]).copy()
        raw["text"] = raw["review"].map(clean_text)
        raw["label"] = raw["sentiment"].str.lower().map({"negative": 0, "positive": 1})
        if raw["label"].isna().any():
            raise ValueError("sentiment must be positive or negative")
        raw = raw[["text", "label"]].drop_duplicates(subset=["text"]).reset_index(drop=True)
        pool, test = train_test_split(raw, test_size=0.2, stratify=raw["label"], random_state=seed)
    if limit is not None:
        if int(limit) < 4: raise ValueError("--limit must be >=4; DEBUG ONLY")
        ntr = min(len(pool), int(limit))
        nte = min(len(test), max(2,int(limit)//2))
        pool,_ = train_test_split(pool,train_size=ntr,stratify=pool.label,random_state=seed)
        test,_ = train_test_split(test,train_size=nte,stratify=test.label,random_state=seed)
    fit,val = train_test_split(pool,test_size=validation_fraction,stratify=pool.label,random_state=seed)
    return [d.reset_index(drop=True).assign(text=lambda x:x.text.map(clean_text)) for d in (fit,val,test)]


def save_split_manifest(train,val,test,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    counts={name:{"rows":len(frame),"labels":{str(k):int(v) for k,v in frame.label.value_counts().items()}} for name,frame in [("train",train),("validation",val),("test",test)]}
    (output_dir/"dataset_manifest.json").write_text(json.dumps(counts,indent=2))
    return counts