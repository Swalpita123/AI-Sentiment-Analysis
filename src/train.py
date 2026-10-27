"""Fine-tune DistilBERT for binary IMDb sentiment classification."""
import argparse
import json
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_utils import load_imdb_dataset

MODEL_NAME = "distilbert/distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=2000,
                        help="Number of training examples; 0 means full train split.")
    parser.add_argument("--eval-samples", type=int, default=1000,
                        help="Number of test examples; 0 means full test split.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--output-dir", default="outputs/sentiment_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(metrics_dir, exist_ok=True)

    print("Loading IMDb dataset...")
    train_ds, test_ds = load_imdb_dataset(args.train_samples, args.eval_samples, args.seed)

    print(f"Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_ds,
    #     eval_dataset=test_ds,
    #     tokenizer=tokenizer,
    #     data_collator=collator,
    #     compute_metrics=compute_metrics,
    # )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )


    print("Starting fine-tuning...")
    train_result = trainer.train()
    print("Evaluating model...")
    evaluation = trainer.evaluate()

    predictions_output = trainer.predict(test_ds)
    predicted_ids = np.argmax(predictions_output.predictions, axis=-1)
    true_ids = predictions_output.label_ids

    report = classification_report(
        true_ids, predicted_ids, target_names=["NEGATIVE", "POSITIVE"],
        output_dict=True, zero_division=0
    )
    with open(os.path.join(metrics_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_result.metrics, "evaluation_metrics": evaluation}, f, indent=2, default=float)

    cm = confusion_matrix(true_ids, predicted_ids)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NEGATIVE", "POSITIVE"], yticklabels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("IMDb Sentiment - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=160)
    plt.close()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("\nTraining complete.")
    print(json.dumps(evaluation, indent=2, default=float))
    print(f"Model saved to: {args.output_dir}")
    print(f"Evaluation artifacts saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
