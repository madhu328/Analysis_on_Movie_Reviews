"""BiLSTM implementation with optional pretrained GloVe vectors."""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from .evaluate import evaluate_predictions

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?.,;:]",re.I)
def tokenize(text): return TOKEN_RE.findall(text.lower())

def build_vocab(texts,max_size=30000):
    counts=Counter(tok for text in texts for tok in tokenize(text))
    return {"<PAD>":0,"<UNK>":1, **{tok:i+2 for i,(tok,_) in enumerate(counts.most_common(max_size-2))}}

def encode(text,vocab,max_length):
    ids=[vocab.get(t,1) for t in tokenize(text)[:max_length]]
    return ids+[0]*(max_length-len(ids))

def load_glove(path,vocab,dim):
    rng=np.random.default_rng(42);matrix=rng.normal(0,0.05,(len(vocab),dim)).astype("float32");matrix[0]=0
    found=0
    with open(path,encoding="utf-8",errors="ignore") as f:
        for line in f:
            pieces=line.rstrip().split(" ")
            if len(pieces)!=dim+1:continue
            word=pieces[0]
            if word in vocab:
                try:matrix[vocab[word]]=np.asarray(pieces[1:],dtype="float32");found+=1
                except ValueError:pass
    return matrix,found

class BiLSTMClassifier:
    """Lazy PyTorch construction avoids requiring torch for preprocessing tests."""
    def __new__(cls, vocab_size, embedding_dim, hidden_dim):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.drop = nn.Dropout(0.3)
                self.fc = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                lengths = (x != 0).sum(1).clamp(min=1).cpu()
                embedded = self.embedding(x)
                packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
                _, (hidden, _) = self.lstm(packed)
                h = torch.cat((hidden[-2], hidden[-1]), dim=1)
                return self.fc(self.drop(h)).squeeze(1)

        return Network()


def train_lstm(train,val,test,out_dir,config,seed=42):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    vocab=build_vocab(train.text,config["vocab_size"])
    (out/"lstm_vocab.json").write_text(json.dumps(vocab))
    length=config["max_length"]
    def loader(frame,shuffle):
        x=torch.tensor([encode(t,vocab,length) for t in frame.text],dtype=torch.long)
        y=torch.tensor(frame.label.to_numpy(),dtype=torch.float32)
        return DataLoader(TensorDataset(x,y),batch_size=config["batch_size"],shuffle=shuffle)
    train_dl,val_dl,test_dl=loader(train,True),loader(val,False),loader(test,False)
    model=BiLSTMClassifier(len(vocab), config["embedding_dim"], config["hidden_dim"]); found=0
    if config.get("pretrained_vectors"):
        weights,found=load_glove(config["pretrained_vectors"],vocab,config["embedding_dim"])
        model.embedding.weight.data.copy_(torch.from_numpy(weights))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
    loss_fn=nn.BCEWithLogitsLoss()
    def predict(dl):
        model.eval(); probs=[]
        with torch.no_grad():
            for x,_ in dl:probs.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
        return np.asarray(probs)
    best=-1;patience=2;no_improve=0;history=[]
    from sklearn.metrics import f1_score
    for epoch in range(config["epochs"]):
        model.train();total=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=float(loss.item())
        val_prob=predict(val_dl);score=f1_score(val.label,(val_prob>=0.5).astype(int))
        history.append({"epoch":epoch+1,"train_loss_sum":total,"val_f1":float(score)})
        if score>best:
            best=score;no_improve=0;torch.save(model.state_dict(),out/"lstm.pt")
        else:
            no_improve+=1
            if no_improve>=patience:break
    model.load_state_dict(torch.load(out/"lstm.pt",map_location=device,weights_only=True))
    prob=predict(test_dl)
    (out/"lstm_config.json").write_text(json.dumps({**config,"vocab_size_actual":len(vocab),"glove_matched":found,"validation_f1":float(best),"history":history},indent=2))
    return evaluate_predictions(test.label,(prob>=0.5).astype(int),test.text,out,"bilstm",prob)