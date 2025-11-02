#!/usr/bin/env python

'''Fine-tune a Transformer (e.g., DeBERTa) with Hugging Face Trainer + Weights & Biases (W&B) support'''

import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DebertaTokenizer
)
import wandb


def create_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tune a Transformer with Trainer + W&B logging")

    # Data / model args
    p.add_argument("--model_name", default="microsoft/deberta-base")
    p.add_argument("--train_file", default="data/train.tsv")
    p.add_argument("--dev_file", default="data/dev.tsv")
    p.add_argument("--max_length", type=int, default=100)

    # Training args
    p.add_argument("--output_dir", default="./out")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="epoch")
    p.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="no")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # WandB
    p.add_argument("--wandb_project", type=str, default="deberta-finetuning",
                   help="Name of the Weights & Biases project")
    p.add_argument("--wandb_log", action="store_true", help="Enable Weights & Biases logging")
    return p.parse_args()


def read_corpus(corpus_file):
    documents, labels = [], []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) >= 2:
                documents.append(tokens[0])
                labels.append(tokens[1])
    return documents, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}


def print_results(metrics):
    acc = round(metrics["eval_accuracy"] * 100, 1)
    f1_micro = round(metrics["eval_f1_micro"] * 100, 1)
    f1_macro = round(metrics["eval_f1_macro"] * 100, 1)
    print("\nFinal metrics:")
    print(f"Accuracy: {acc}")
    print(f"Micro F1: {f1_micro}")
    print(f"Macro F1: {f1_macro}")


def prepare_data(args, tokenizer):
    print(f"Reading data from {args.train_file} and {args.dev_file}...")
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    le = LabelEncoder()
    y_train_ids = le.fit_transform(Y_train).astype(np.int64)
    y_dev_ids = le.transform(Y_dev).astype(np.int64)

    tok_train = tokenizer(X_train, padding=True, truncation=True, max_length=args.max_length)
    tok_dev = tokenizer(X_dev, padding=True, truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_dict({**tok_train, "labels": y_train_ids}).with_format("torch")
    dev_ds = Dataset.from_dict({**tok_dev, "labels": y_dev_ids}).with_format("torch")

    return train_ds, dev_ds, le


def main():
    args = create_arg_parser()

    # ✅ Initialize Weights & Biases (optional)
    if args.wandb_log:
        wandb.init(project=args.wandb_project, config=vars(args))
        print(f"✅ W&B logging enabled — Project: {args.wandb_project}")
    else:
        # disable wandb if not requested
        import os
        os.environ["WANDB_DISABLED"] = "true"

    print("Training model:", args.model_name)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Data
    train_ds, dev_ds, le = prepare_data(args, tokenizer)
    num_labels = len(le.classes_)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        do_eval=True,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        report_to="wandb" if args.wandb_log else [],
        load_best_model_at_end=True if args.save_strategy != "no" else False,
    )

    print("Evaluation strategy:", training_args.eval_strategy)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )

    print(tokenizer.vocab_size)
    print(model.config.vocab_size)

    trainer.train()
    metrics = trainer.evaluate()
    print_results(metrics)

    print("\nPerforming error analysis...")
    preds_output = trainer.predict(dev_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    for i, class_name in enumerate(le.classes_):
        total = np.sum(labels == i)
        correct = np.sum((labels == i) & (preds == i))
        incorrect = total - correct
        acc = (correct / total * 100) if total > 0 else 0
        print(f"Class '{class_name}': Total={total}, Correct={correct}, Incorrect={incorrect}, Accuracy={acc:.2f}%")

    if args.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()



