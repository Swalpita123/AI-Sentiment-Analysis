# Deep Learning Classification & Text Processing Pipeline

An end-to-end NLP portfolio project that classifies movie reviews as **positive** or **negative**. It compares a traditional machine-learning baseline (TF-IDF + Logistic Regression) with a fine-tuned DistilBERT transformer.

## Features

- Public IMDb review dataset downloaded through Hugging Face Datasets
- Text normalization and tokenization
- Reproducible train/test sampling with a fixed seed
- TF-IDF + Logistic Regression baseline
- DistilBERT fine-tuning using PyTorch and Hugging Face Transformers
- Accuracy, precision, recall, F1-score, classification report, and confusion matrix
- CLI prediction on new text
- Saves model/tokenizer and evaluation artifacts locally

> **Important:** The repository contains the training pipeline, not a pretrained fine-tuned model. You must run training before using `src/predict.py`. Report only the metrics produced by your own run; do not claim unmeasured accuracy.

## 1. Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended
- VS Code and its Python extension
- Internet connection for downloading packages, the IMDb dataset, and the pretrained DistilBERT model
- RAM: 8 GB minimum recommended; 16 GB is more comfortable
- GPU is optional. CPU training works but can be slow. A CUDA-compatible PyTorch setup can speed it up.

## 2. Create and activate a virtual environment

Open this folder in VS Code. In the VS Code terminal (PowerShell), run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.12 is not installed, install it from https://www.python.org/downloads/ and reopen VS Code. You can also use `py -0p` to list installed Python versions.

## 3. Train a traditional baseline (recommended)

This is quicker than transformer training and provides a point of comparison:

```powershell
python -m src.baseline --train-samples 2000 --eval-samples 1000
```

Metrics are saved in `outputs/baseline/metrics.json`.

For the full IMDb training/test splits, use `--train-samples 0 --eval-samples 0`. This takes more memory/time.

## 4. Fine-tune DistilBERT

Start with the small run to verify your environment:

```powershell
python -m src.train --train-samples 500 --eval-samples 250 --epochs 1 --batch-size 4
```

Then run a more meaningful experiment:

```powershell
python -m src.train --train-samples 2000 --eval-samples 1000 --epochs 1 --batch-size 8
```

Use `--train-samples 0 --eval-samples 0` to use all 25,000 training and 25,000 test reviews. Full fine-tuning can take hours on CPU; use a GPU if available. If you run out of memory, reduce `--batch-size` to 2 or `--max-length` to 128.

The fine-tuned model is saved in `outputs/sentiment_model/`. Evaluation artifacts are in `outputs/sentiment_model/evaluation/`:
- `metrics.json`
- `classification_report.json`
- `confusion_matrix.png`

## 5. Predict sentiment

After training succeeds:

```powershell
python -m src.predict --text "The acting was wonderful and the story was engaging."
python -m src.predict
```

The second command opens interactive mode. Type `quit` to exit.

Example output format (your confidence will vary):

```text
Sentiment: POSITIVE | confidence: 94.20%
```

The number above is an illustration only, not a guaranteed result.

## 6. Project structure

```text
text_classification_pipeline/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── baseline.py
    ├── train.py
    └── predict.py
```

Generated datasets/cache and model files are not committed to Git by default. Hugging Face caches datasets/models in its local cache; the fine-tuned model is saved under `outputs/`.

## 7. How it works

1. **Load data:** IMDb contains movie reviews with binary labels (0 = negative, 1 = positive).
2. **Normalize:** remove HTML line-break tags and normalize whitespace.
3. **Tokenize:** DistilBERT's tokenizer maps text to token IDs and attention masks.
4. **Fine-tune:** the pretrained transformer learns the sentiment task from labeled reviews.
5. **Evaluate:** compute accuracy, precision, recall, F1 and confusion matrix on the test split.
6. **Predict:** tokenize new user text and choose the label with the highest softmax probability.

The baseline uses TF-IDF word/phrase features and Logistic Regression. Compare its metrics with DistilBERT's metrics using the same sample sizes for a fair comparison.

## 8. Common Windows fixes

- **`py` cannot find Python 3.12:** install Python 3.12 or use your installed supported version when creating the environment.
- **`ModuleNotFoundError: src`:** run commands from the project root and use `python -m src.train`, not `python src/train.py`.
- **PowerShell activation blocked:** use the process-scoped execution-policy command in Step 2.
- **CUDA/GPU issue:** start with CPU; PyTorch will use CUDA automatically when it is available.
- **Download interrupted:** rerun the same command. Hugging Face downloads are cached and usually resume/reuse completed files.
- **Out of memory:** lower batch size and max length; use fewer samples while testing.

## 9. Interview explanation

“I built an end-to-end sentiment classification pipeline using Python, Pandas, Scikit-learn, PyTorch, and Hugging Face Transformers. I prepared labeled IMDb reviews, normalized text, and created a TF-IDF plus Logistic Regression baseline. I then fine-tuned DistilBERT for binary sentiment classification and evaluated it using accuracy, precision, recall, F1-score, and a confusion matrix. I also saved the model and tokenizer so that new reviews can be classified through a command-line prediction script.”

Only say that you fine-tuned and evaluated the model after successfully running the training script. Be ready to discuss your actual dataset size, training time, hardware, and measured metrics.
