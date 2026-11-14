"""Fine-tune DistilBERT with validation-only model selection."""
import json
from pathlib import Path
import numpy as np
from .evaluate import evaluate_predictions

def train_transformer(train,val,test,out_dir,config,seed=42):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    model_name=config["name"]
    tok=AutoTokenizer.from_pretrained(model_name)
    def as_ds(frame):
        ds=Dataset.from_dict({"text":frame.text.tolist(),"labels":frame.label.astype(int).tolist()})
        return ds.map(lambda batch:tok(batch["text"],truncation=True,max_length=config["max_length"]),batched=True,remove_columns=["text"])
    tr,va,te=as_ds(train),as_ds(val),as_ds(test)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
    kwargs=dict(output_dir=str(out/"transformer_checkpoints"),learning_rate=config["learning_rate"],per_device_train_batch_size=config["batch_size"],per_device_eval_batch_size=config["batch_size"],num_train_epochs=config["epochs"],weight_decay=0.01,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="f1",greater_is_better=True,save_total_limit=1,seed=seed,report_to="none")
    args=TrainingArguments(**kwargs)
    from sklearn.metrics import precision_recall_fscore_support,accuracy_score
    def metric(x):
        predicted=np.argmax(x.predictions,axis=-1)
        pr,rc,f1,_=precision_recall_fscore_support(x.label_ids,predicted,average="binary",zero_division=0)
        return {"f1":float(f1),"accuracy":float(accuracy_score(x.label_ids,predicted)),"precision":float(pr),"recall":float(rc)}
    trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,processing_class=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metric)
    trainer.train(); trainer.save_model(str(out/"distilbert"));tok.save_pretrained(str(out/"distilbert"))
    preds=trainer.predict(te).predictions
    probabilities=torch.softmax(torch.tensor(preds),dim=-1)[:,1].numpy()
    result=evaluate_predictions(test.label,np.argmax(preds,axis=1),test.text,out,"distilbert",probabilities)
    (out/"transformer_config.json").write_text(json.dumps(config,indent=2))
    return result