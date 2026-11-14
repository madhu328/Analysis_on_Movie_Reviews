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


def test_kaggle_loader_prevents_duplicate_leakage(tmp_path):
    import pandas as pd
    from movie_sentiment.data import load_imdb
    rows=[]
    for i in range(120):
        label='positive' if i%2 else 'negative'
        rows.append({'review':f'Unique movie review number {i}', 'sentiment': label})
    rows.append(rows[0].copy())
    p=tmp_path/'tiny.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    train,val,test=load_imdb(source=str(p),seed=42)
    assert len(train)+len(val)+len(test)==120
    assert len(set(train.text)&set(val.text))==0
    assert len(set(train.text)&set(test.text))==0
    assert len(set(val.text)&set(test.text))==0