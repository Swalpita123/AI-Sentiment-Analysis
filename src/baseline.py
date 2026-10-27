"""Train a TF-IDF + Logistic Regression baseline for comparison."""
import argparse
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from src.data_utils import load_imdb_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/baseline")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, test_ds = load_imdb_dataset(args.train_samples, args.eval_samples)
    x_train, y_train = train_ds["text"], train_ds["label"]
    x_test, y_test = test_ds["text"], test_ds["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    print("Training TF-IDF + Logistic Regression baseline...")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test, predictions, target_names=["NEGATIVE", "POSITIVE"], output_dict=True, zero_division=0
        )
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))
    print(f"Metrics saved to {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
