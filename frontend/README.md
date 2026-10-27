# AI-Powered Movie Review Sentiment Analysis

An AI-powered text classification project that analyzes IMDb movie reviews and predicts whether a review expresses a **positive or negative sentiment**. The project uses a fine-tuned DistilBERT transformer model and provides predictions through a FastAPI backend and an interactive React frontend.

## Features

- Sentiment classification of movie reviews as Positive or Negative.
- DistilBERT-based natural language processing.
- Confidence scores for predictions.
- FastAPI REST API for model inference.
- Interactive React user interface.
- Model evaluation using Accuracy, Precision, Recall, and F1-score.

## Tech Stack

Python, PyTorch, DistilBERT, Hugging Face Transformers, FastAPI, Uvicorn, React.js, JavaScript (ES6+), HTML5, CSS3, REST APIs, IMDb Dataset, Git, GitHub.

## Project Structure

```text
text_classification_pipeline/
├── src/
│   ├── data_utils.py
│   ├── baseline.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── index.html
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Performance

The model was trained and evaluated on a sample of the IMDb movie review dataset.

- **Evaluation Accuracy:** 82.8%
- **Evaluation Samples:** 250
- **Model:** DistilBERT

*Note: These results are from a limited evaluation sample and may differ when evaluated on a larger dataset.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-sentiment-analysis.git
cd ai-sentiment-analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Set up the Python environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train --train-samples 500 --eval-samples 250 --epochs 1 --batch-size 4
```

Training generates the model files in `outputs/sentiment_model/`. The training script downloads the required dataset and pretrained model as needed.

### 4. Start the FastAPI backend

From the project root, run:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### 5. Start the React frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed in the terminal, usually:

http://localhost:5173

Keep both the backend and frontend running while using the application.

## API Usage

**Endpoint:** `POST /predict`

Example request:

```json
{
  "text": "The movie was excellent and the acting was amazing."
}
```

The API returns the predicted sentiment, confidence score, and original review text.

## Future Improvements

- Train and evaluate the model on a larger dataset.
- Improve prediction performance on mixed or ambiguous reviews.
- Deploy the frontend and backend for public access.
- Add support for additional text classification tasks.

## Author

Developed as an AI/NLP project to demonstrate transformer-based sentiment analysis, model evaluation, REST API development, and full-stack integration.
