"""Run sentiment predictions with a locally fine-tuned model."""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict(text: str, model_dir: str = "outputs/sentiment_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    labels = model.config.id2label
    label = labels.get(idx, labels.get(str(idx), "POSITIVE" if idx else "NEGATIVE"))
    return label, float(probs[idx].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Text to classify. If omitted, interactive mode starts.")
    parser.add_argument("--model-dir", default="outputs/sentiment_model")
    args = parser.parse_args()

    if args.text:
        label, confidence = predict(args.text, args.model_dir)
        print(f"Sentiment: {label} | confidence: {confidence:.2%}")
        return

    print("IMDb Sentiment Classifier (type 'quit' to exit)")
    while True:
        text = input("\nEnter a review: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            print("Please enter some text.")
            continue
        label, confidence = predict(text, args.model_dir)
        print(f"Sentiment: {label} | confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
