"""Metrics, confusion matrices, errors, model comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred, text, out_dir, name, probabilities=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result = {"model": name, "accuracy": float(accuracy_score(y_true,y_pred)), "precision":float(precision), "recall":float(recall), "f1":float(f1), "samples":int(len(y_true)), "positive_class":1}
    (out / f"{name}_metrics.json").write_text(json.dumps(result,indent=2))
    (out / f"{name}_classification_report.txt").write_text(classification_report(y_true,y_pred,labels=[0,1],target_names=["negative","positive"],zero_division=0))
    cm = confusion_matrix(y_true,y_pred,labels=[0,1]); np.savetxt(out / f"{name}_confusion_matrix.csv",cm,fmt="%d",delimiter=",")
    rows = pd.DataFrame({"text": list(text),"true_label":y_true,"predicted_label":y_pred})
    if probabilities is not None: rows["positive_probability"] = probabilities
    rows.loc[rows.true_label!=rows.predicted_label].to_csv(out / f"{name}_errors.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(4.8,4)); ax.imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","Positive"],yticklabels=["Negative","Positive"],xlabel="Predicted",ylabel="Actual",title=f"{name} confusion matrix")
    fig.tight_layout(); fig.savefig(out/f"{name}_confusion_matrix.png",dpi=160); plt.close(fig)
    return result


def comparison(out_dir):
    out=Path(out_dir); records=[]
    for f in sorted(out.glob("*_metrics.json")):
        records.append(json.loads(f.read_text()))
    if not records: raise FileNotFoundError("No evaluation metrics yet; run training/evaluation first.")
    frame=pd.DataFrame(records); frame.to_csv(out / "model_comparison.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    frame.set_index("model")[["accuracy","precision","recall","f1"]].plot.bar(ax=ax,ylim=(0,1))
    ax.set_ylabel("Score");ax.set_xlabel("");fig.tight_layout();fig.savefig(out/"model_comparison.png",dpi=160);plt.close(fig)
    return frame