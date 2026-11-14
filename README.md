# Analysis on Movie Reviews Using Traditional NLP and Deep Learning

A reproducible NLP course project comparing **TF-IDF + Logistic Regression**, **BiLSTM with optional pretrained GloVe embeddings**, and fine-tuned **DistilBERT** for IMDb binary sentiment classification.

> **Status:** Training and held-out evaluation completed. The real results, figures, error-analysis CSVs, and final PDF report are included in `results/` and `report/`. Trained weights are backed up separately to Google Drive, not included in this repository.

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

**Dataset:** Download `IMDB Dataset.csv` from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews and place at `data/IMDB Dataset.csv`. The Kaggle loader removes duplicate review text before creating stratified training/validation/test splits. Actual split: 35,695 train, 3,967 validation, 9,916 test (49,578 total). Seed 42. For the optional original Stanford split, change `dataset` in YAML to `stanfordnlp/imdb`.

## Training and evaluation (completed experiment; rerun commands below)

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

All models use the same held-out Kaggle test partition, seed 42, and training/validation allocation. Logistic Regression chooses C from 0.1, 1 and 10 by validation F1. BiLSTM chooses the best epoch by validation F1 and stops on patience 2. DistilBERT selects the best saved epoch by validation F1. Scores reported are positive-class precision, recall and F1 (threshold 0.5), plus accuracy; see classification reports and confusion matrices. No training on held-out test examples. Repeated tuning after test evaluation would compromise held-out validity.

## Error analysis

`imdb-sentiment errors` tags each model's misclassifications using explainable **heuristics** for negation, contrast, mixed sentiment and sarcasm *candidates*. These flags are not proof of sarcasm; manual inspection is necessary. All errors remain in CSV for qualitative analysis.

## Interactive demo

Once trained: `streamlit run app.py`. The app discovers available trained checkpoints in `artifacts/`; restore trained weights before launching the demo.

## Files

- `src/movie_sentiment/`: dataset, preprocessing, EDA, models, evaluation, errors, inference, CLI
- `configs/default.yaml`: reproducibility and hyperparameters
- `tests/`: fast unit tests (no external dataset needed)
- `report/Project_Report.pdf`: final report with actual measured results
- `results/`: measured metric files, plots and error-analysis CSVs
- `scripts/build_submission.py`: create single ZIP with source and PDF

## Submission

When ready: `python scripts/build_submission.py`. This creates `submission/IMDb_Movie_Reviews_NLP_Submission.zip` (relative to the repo root), containing source and `report/Project_Report.pdf`; model weights and datasets are excluded. Before submission verify all team registration numbers. The report already includes actual metrics and figures.

## License and dataset

IMDb reviews dataset from Stanford / Maas et al. (ACL 2011); DistilBERT: Sanh et al. (2019). Refer to model/dataset licenses and acknowledge their sources. This student project does not claim novel model architectures.


## Verified actual results (Kaggle IMDb test set: n=9,916)

| Model | Accuracy | Precision | Recall | Positive F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 90.81% | 90.31% | 91.52% | 90.91% |
| BiLSTM | 87.74% | 86.88% | 89.01% | 87.93% |
| DistilBERT | **92.16%** | **91.70%** | **92.79%** | **92.24%** |

The `results/` folder contains the measured comparison, confusion matrices, metric JSON files and tagged misclassification CSVs. Sarcasm/contrast/negation tags are heuristic. The test partition was held out from fitting and model selection.

### Running the trained demo without retraining

1. Install: `pip install -e ".[deep,demo,test]"`.
2. Restore the trained Google Drive backup folder `IMDb_NLP_Artifacts` to project-relative `artifacts/`. Model `.pt`, `.joblib`, and Transformer model weights must be present in this folder to enable predictions.
3. Run `imdb-sentiment compare`, `imdb-sentiment errors`, and `streamlit run app.py`.
4. On Colab, the command can disappear after a runtime reset; reclone the repository and repeat `pip install -e ".[deep,demo,test]"`, then restore `artifacts/` before running inference.

### Final submission

The source ZIP includes scripts, the final report PDF and results/ with the measured charts, comparisons, and errors. It excludes the raw Kaggle CSV and trained model weights due to size.