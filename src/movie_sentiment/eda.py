"""Descriptive EDA on training data only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_eda(train,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    counts=train.label.map({0:"Negative",1:"Positive"}).value_counts().reindex(["Negative","Positive"])
    counts.plot.bar(ax=axes[0]);axes[0].set(title="Training sentiment distribution",ylabel="Reviews",xlabel="")
    train.text.str.split().str.len().clip(upper=1000).hist(ax=axes[1],bins=40)
    axes[1].set(title="Training review length (clipped at 1000 words)",xlabel="Words",ylabel="Reviews")
    fig.tight_layout();fig.savefig(out/"eda.png",dpi=160);plt.close(fig)