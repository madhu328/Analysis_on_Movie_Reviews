"""Command line interface for training and evaluation."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import yaml

def main():
    parser=argparse.ArgumentParser(description="IMDb sentiment project")
    parser.add_argument("command",choices=["eda","baseline","lstm","transformer","compare","errors","all","predict"])
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--limit",type=int,default=None,help="DEBUG ONLY: reduced data; not final metrics")
    parser.add_argument("--text",default=None,help="Review text for predict command")
    parser.add_argument("--model",choices=["baseline","lstm","transformer"],default="baseline")
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    out=Path(cfg["output_dir"]);out.mkdir(exist_ok=True,parents=True)
    if args.command == "predict":
        if not args.text:
            parser.error("predict requires --text 'your movie review'")
        from .predict import predict_baseline, predict_lstm, predict_transformer
        functions = {"baseline": predict_baseline, "lstm": predict_lstm, "transformer": predict_transformer}
        print(json.dumps(functions[args.model](args.text, out), indent=2))
        return
    if args.command in ("compare","errors"):
        from .evaluate import comparison
        from .analysis import analyze_errors
        result=comparison(out) if args.command=="compare" else analyze_errors(out)
        print(result.to_string(index=False));return
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"])
    from .data import load_imdb,save_split_manifest
    train,val,test=load_imdb(cfg["seed"],cfg["validation_fraction"],args.limit)
    manifest=save_split_manifest(train,val,test,out)
    (out/"run_metadata.json").write_text(json.dumps({"seed":cfg["seed"],"debug_limit":args.limit,"dataset":cfg["dataset"],"split":manifest},indent=2))
    if args.command in ("eda","all"):
        from .eda import make_eda
        make_eda(train,out)
    if args.command in ("baseline","all"):
        from .baseline import train_baseline
        print(train_baseline(train,val,test,out,cfg["max_features"],cfg["ngram_max"],cfg["seed"]))
    if args.command in ("lstm","all"):
        from .lstm import train_lstm
        print(train_lstm(train,val,test,out,cfg["lstm"],cfg["seed"]))
    if args.command in ("transformer","all"):
        from .transformer import train_transformer
        print(train_transformer(train,val,test,out,cfg["transformer"],cfg["seed"]))
    if args.command=="all":
        from .evaluate import comparison
        from .analysis import analyze_errors
        print(comparison(out));print(analyze_errors(out))

if __name__=="__main__":main()