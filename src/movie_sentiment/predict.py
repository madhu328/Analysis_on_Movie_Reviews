"""Load persisted model checkpoints and classify a new review."""
import json
from pathlib import Path


def _result(p):
    p = float(p)
    return {"label": "positive" if p >= 0.5 else "negative", "positive_probability": p}


def predict_baseline(text, artifact_dir="artifacts"):
    import joblib
    model = joblib.load(Path(artifact_dir) / "baseline.joblib")
    return _result(model.predict_proba([text])[0, 1])


def predict_lstm(text, artifact_dir="artifacts"):
    """Recreate the BiLSTM and load its best checkpoint; CPU is sufficient for inference."""
    import torch
    from .lstm import BiLSTMClassifier, encode
    out = Path(artifact_dir)
    config = json.loads((out / "lstm_config.json").read_text())
    vocab = json.loads((out / "lstm_vocab.json").read_text())
    model = BiLSTMClassifier(len(vocab), config["embedding_dim"], config["hidden_dim"])
    model.load_state_dict(torch.load(out / "lstm.pt", map_location="cpu", weights_only=True))
    model.eval()
    tokens = torch.tensor([encode(text, vocab, config["max_length"])], dtype=torch.long)
    with torch.no_grad():
        p = torch.sigmoid(model(tokens)).item()
    return _result(p)


def predict_transformer(text, artifact_dir="artifacts"):
    import torch
    from transformers import pipeline
    path = str(Path(artifact_dir) / "distilbert")
    classifier = pipeline("text-classification", model=path, tokenizer=path,
                          device=0 if torch.cuda.is_available() else -1,
                          truncation=True, max_length=256)
    result = classifier(text)[0]
    p = result["score"] if result["label"].upper() in ("LABEL_1", "POSITIVE") else 1 - result["score"]
    return _result(p)


def available_models(artifact_dir="artifacts"):
    out = Path(artifact_dir)
    models = {}
    if (out / "baseline.joblib").exists():
        models["TF-IDF + Logistic Regression"] = predict_baseline
    if all((out / x).exists() for x in ("lstm.pt", "lstm_config.json", "lstm_vocab.json")):
        models["BiLSTM"] = predict_lstm
    if (out / "distilbert" / "config.json").exists():
        models["DistilBERT"] = predict_transformer
    return models