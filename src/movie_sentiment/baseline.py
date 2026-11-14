"""TF-IDF + Logistic Regression; fit ONLY on training partition."""
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from .evaluate import evaluate_predictions


def train_baseline(train,val,test,output_dir,max_features=50000,ngram_max=2,seed=42):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    variants={"C_0.1":0.1,"C_1":1.0,"C_10":10.0}
    best=None; best_score=-1
    for key,C in variants.items():
        model=Pipeline([("tfidf",TfidfVectorizer(max_features=max_features,ngram_range=(1,ngram_max),sublinear_tf=True,strip_accents="unicode")),("logreg",LogisticRegression(C=C,max_iter=1500,random_state=seed))])
        model.fit(train.text,train.label)
        score=f1_score(val.label,model.predict(val.text))
        if score>best_score: best,best_score,best_c=model,float(score),C
    joblib.dump(best,out/"baseline.joblib")
    pred=best.predict(test.text); prob=best.predict_proba(test.text)[:,1]
    result=evaluate_predictions(test.label,pred,test.text,out,"tfidf_logreg",prob)
    result["validation_f1"]=best_score;result["selected_C"]=best_c
    import json
    (out/"baseline_training.json").write_text(json.dumps({"validation_f1":best_score,"selected_C":best_c,"hyperparameter_candidates":list(variants.values())},indent=2))
    return result