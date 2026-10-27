from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Locate the trained model from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "sentiment_model"

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict positive or negative sentiment using DistilBERT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once when the API starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval()


class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Movie review text to classify",
    )


@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# @app.post("/predict")
# def predict_sentiment(request: ReviewRequest):
#     text = request.text.strip()

#     if not text:
#         raise HTTPException(
#             status_code=400,
#             detail="Review text cannot be empty.",
#         )

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=256,
#     )

#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probabilities = torch.softmax(outputs.logits, dim=1)[0]

#     predicted_id = int(torch.argmax(probabilities).item())
#     confidence = float(probabilities[predicted_id].item())

#     # IMDb dataset labels: 0 = negative, 1 = positive
#     sentiment = "POSITIVE" if predicted_id == 1 else "NEGATIVE"

#     return {
#         "sentiment": sentiment,
#         "confidence": round(confidence * 100, 2),
#         "text": text,
#     }


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Review text cannot be empty.",
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    # IMDb labels: 0 = negative, 1 = positive
    sentiment = "POSITIVE" if predicted_id == 1 else "NEGATIVE"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "text": text,
    }
