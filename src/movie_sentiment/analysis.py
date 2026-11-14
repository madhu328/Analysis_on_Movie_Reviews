"""Heuristic audit of model mistakes; categories can overlap."""
import re
from pathlib import Path
import pandas as pd

PATTERNS={
    "negation":r"\b(?:not|no|never|n't|neither|nor|without|hardly)\b",
    "contrast":r"\b(?:but|however|although|though|yet|despite|whereas)\b",
    "sarcasm_candidate":r"\b(?:yeah right|as if|just what i needed|what a masterpiece|surely|obviously)\b|[!?]{2,}",
    "mixed_sentiment":r"\b(?:good|great|excellent|love)\b.*\b(?:bad|awful|boring|hate)\b|\b(?:bad|awful|boring|hate)\b.*\b(?:good|great|excellent|love)\b",
}

def analyze_errors(out_dir):
    out=Path(out_dir)
    summary=[]
    for f in sorted(out.glob("*_errors.csv")):
        frame=pd.read_csv(f)
        for label,pattern in PATTERNS.items():
            frame[label]=frame.text.fillna("").str.contains(pattern,flags=re.I,regex=True)
            summary.append({"model":f.stem.removesuffix("_errors"),"category":label,"errors_matching":int(frame[label].sum()),"total_errors":len(frame)})
        frame.to_csv(out/f"{f.stem}_tagged.csv",index=False)
    if not summary:raise FileNotFoundError("Run model evaluations first.")
    result=pd.DataFrame(summary);result.to_csv(out/"error_analysis_summary.csv",index=False)
    return result