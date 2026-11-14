# Analysis on Movie Reviews Using Traditional NLP and Deep Learning

A reproducible NLP course project comparing **TF-IDF + Logistic Regression**, **BiLSTM with optional pretrained GloVe embeddings**, and fine-tuned **DistilBERT** for IMDb binary sentiment classification.

> **Status:** Source code and report drafted; full model training and real evaluation are **pending**. Do not present any scores or comparisons as measured until you run experiments. No pretrained model files are distributed here.

## Group members

| Name | Registration number |
| --- | --- |
| Maloth Madhu | 12341370 |
| Bhukya Raju | 1234052 |
| Maloth Srivastav Aditya Naik | 12240960 |
| Amarjeet Kumar | 12240110 |

*Registration numbers are reproduced exactly as supplied; verify before submission.*

## Setup

Python 3.10+; **GPU recommended** for neural models. Run commands from repository root.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[deep,test,demo]'
pytest -q
```

The IMDb dataset is downloaded automatically from Hugging Face (`stanfordnlp/imdb`) on first run (requires internet access). Train partition: official 25,000; validation split: 10% of official training partition (2,500); fit: 22,500. **Official test partition: 25,000**, untouched until evaluation.

## Training and evaluation (planned for a later session)

```bash
imdb-sentiment eda
imdb-sentiment baseline
imdb-sentiment lstm
imdb-sentiment transformer
imdb-sentiment compare
imdb-sentiment errors
```

Or `imdb-sentiment all`. Neural training requires `pip install -e '.[deep]'`. The LSTM is **randomly initialized by default**; to use *pretrained* word vectors, download a trusted GloVe text file separately and set `lstm.pretrained_vectors` and matching `lstm.embedding_dim` in `configs/default.yaml`. For example GloVe 6B 100d requires `embedding_dim: 100`. Pretrained embeddings are not bundled due to size and licensing. We recommend using GloVe for the final comparison.

To run a quick **debug-only** subset: `imdb-sentiment baseline --limit 1000`. Such results must **not** be reported as final project metrics. For final reporting run without `--limit` and save the generated `artifacts/` results. Models and large downloads are intentionally excluded from Git.

## Evaluation protocol

All models use the same official test set, seed 42 and training/validation allocation. Logistic Regression chooses C from 0.1, 1 and 10 by validation F1. BiLSTM chooses the best epoch by validation F1 and stops on patience 2. DistilBERT selects the best saved epoch by validation F1. Scores reported are positive-class precision, recall and F1 (threshold 0.5), plus accuracy; see classification reports and confusion matrices. No training on official test examples. Repeated tuning after test evaluation would compromise held-out validity.

## Error analysis

`imdb-sentiment errors` tags each model's misclassifications using explainable **heuristics** for negation, contrast, mixed sentiment and sarcasm *candidates*. These flags are not proof of sarcasm; manual inspection is necessary. All errors remain in CSV for qualitative analysis.

## Interactive demo

Once trained: `streamlit run app.py`. The app automatically offers baseline and/or transformer models if artifacts exist; BiLSTM evaluation is supported in the CLI and can be extended for demo inference.

## Files

- `src/movie_sentiment/`: dataset, preprocessing, EDA, models, evaluation, errors, inference, CLI
- `configs/default.yaml`: reproducibility and hyperparameters
- `tests/`: fast unit tests (no external dataset needed)
- `report/Project_Report.pdf`: project report with honest pending-results status
- `results/README.md`: instructions for adding measured results
- `scripts/build_submission.py`: create single ZIP with source and PDF

## Submission

When ready: `python scripts/build_submission.py`. This creates `submission/IMDb_Movie_Reviews_NLP_Submission.zip` (relative to the repo root), containing source and `report/Project_Report.pdf`; model weights and datasets are excluded. Before submission verify all team registration numbers and replace pending-results text with actual metrics/figures after experiments.

## License and dataset

IMDb reviews dataset from Stanford / Maas et al. (ACL 2011); DistilBERT: Sanh et al. (2019). Refer to model/dataset licenses and acknowledge their sources. This student project does not claim novel model architectures.

## Step-by-step: Google Colab training (recommended)

1. Open a new notebook at https://colab.research.google.com and select **Runtime > Change runtime type > T4 GPU** (if available). For the baseline, CPU is sufficient.
2. In a notebook cell, clone the public repository and install its packages:

   ```python
   !git clone https://github.com/madhu328/Analysis_on_Movie_Reviews.git
   %cd Analysis_on_Movie_Reviews
   !pip install -e '.[deep,demo,test]'
   ```

3. Run the training stages separately. **You must actually execute these cells**; downloading source code does not train the models.

   ```python
   !imdb-sentiment eda
   !imdb-sentiment baseline
   !imdb-sentiment lstm
   !imdb-sentiment transformer
   !imdb-sentiment compare
   !imdb-sentiment errors
   ```

   All commands use `configs/default.yaml`. A GPU is strongly recommended for the LSTM and especially DistilBERT. Actual runtime depends on hardware and dataset download speed. By default the LSTM trains embeddings from scratch; for the specified word-vector comparison, supply an independently downloaded **GloVe 100-dimensional** text file and update `lstm.pretrained_vectors` in the YAML before LSTM training.

4. **Keep the generated `artifacts/` directory.** It contains trained checkpoints, test metrics, confusion matrices, error samples and graphs. Colab storage is temporary; download or copy it to Google Drive before the runtime disconnects.
5. After training, run `!imdb-sentiment compare` and inspect `artifacts/model_comparison.csv` and `artifacts/model_comparison.png`. Copy only the **measured** scores and figures into the final PDF report before course submission. Generate a new ZIP via `!python scripts/build_submission.py` and download it from `submission/`.

**Avoid re-running the entire training procedure just to launch the app.** Train once, preserve the `artifacts/` folder and reuse it for prediction.

## Step-by-step: run on Windows

Open PowerShell inside the cloned repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[deep,demo,test]"
pytest -q
imdb-sentiment baseline
imdb-sentiment predict --model baseline --text "The film was surprisingly good!"
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). After LSTM/Transformer training, the demo automatically detects their checkpoints and shows all available models.

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -e ".[deep,demo,test]"` and `.venv\Scripts\python.exe -m streamlit run app.py`, or use Command Prompt activation `.venv\Scripts\activate.bat`.

## Commands / final outputs

| Command | Expected output after successful execution |
| --- | --- |
| `imdb-sentiment eda` | `artifacts/eda.png` |
| `imdb-sentiment baseline` | `artifacts/baseline.joblib`, TF-IDF metrics and errors |
| `imdb-sentiment lstm` | `artifacts/lstm.pt`, vocabulary, config, BiLSTM metrics |
| `imdb-sentiment transformer` | `artifacts/distilbert/`, DistilBERT metrics |
| `imdb-sentiment compare` | `artifacts/model_comparison.csv` and comparison chart |
| `imdb-sentiment errors` | tagged error CSV files and heuristic summary |
| `imdb-sentiment predict --model lstm --text "great film"` | JSON with sentiment label and positive probability |
| `streamlit run app.py` | interactive web UI displaying predictions and model comparison |

For transformer prediction: `imdb-sentiment predict --model transformer --text "An excellent movie"`. For baseline use `--model baseline`. Predictions require their matching trained model checkpoints.

## Before submitting on 16 November 2026, 5:00 PM IST

1. Confirm all four registration numbers (especially Raju's number as provided).
2. Train all three models without `--limit`, and save the original metric JSON/CSV and figures.
3. Update the report with the actual measured metrics and representative error examples, rather than leaving the "pending" section.
4. Build the ZIP with `python scripts/build_submission.py`, unzip once to verify it includes source and `report/Project_Report.pdf`, and submit **one ZIP per group**.

**Not included in the Git repository by default:** datasets, pretrained word-vector archives, model weights/checkpoints and cached downloads (they can be very large). The `artifacts/` directory is generated by training; its absence before training is expected.