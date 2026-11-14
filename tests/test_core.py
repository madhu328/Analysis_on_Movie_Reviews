import pandas as pd
from movie_sentiment.data import clean_text
from movie_sentiment.lstm import tokenize,build_vocab,encode
from movie_sentiment.evaluate import evaluate_predictions
from movie_sentiment.analysis import analyze_errors

def test_clean_preserves_negation():
    assert clean_text("<b>Not</b> good!<br>Really") == "Not good! Really"

def test_vocab_padding_and_oov():
    v=build_vocab(["Good film", "not good"],max_size=6)
    assert encode("mystery",v,4)==[1,0,0,0]
    assert tokenize("It's NOT good!")==["it's","not","good","!"]

def test_evaluation_and_error_patterns(tmp_path):
    score=evaluate_predictions([1,0,1],[1,1,0],["great movie","not good","yeah right!!"],tmp_path,"toy",[.8,.7,.2])
    assert score["accuracy"]==1/3
    errors=pd.read_csv(tmp_path/"toy_errors.csv");assert len(errors)==2
    summary=analyze_errors(tmp_path)
    assert len(summary)==4
    assert (tmp_path/"toy_confusion_matrix.png").exists()

def test_no_checkpoints_available(tmp_path):
    from movie_sentiment.predict import available_models
    assert available_models(tmp_path) == {}

def test_baseline_inference(tmp_path):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from movie_sentiment.predict import predict_baseline, available_models
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(["awful terrible film", "horrible boring story", "wonderful great movie", "excellent brilliant film"], [0, 0, 1, 1])
    joblib.dump(model, tmp_path / "baseline.joblib")
    result = predict_baseline("great movie", tmp_path)
    assert result["label"] in ("positive", "negative")
    assert 0 <= result["positive_probability"] <= 1
    assert len(available_models(tmp_path)) == 1