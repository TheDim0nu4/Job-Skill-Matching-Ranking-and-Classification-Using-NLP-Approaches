"""Навчання LogisticRegression на embeddings та оцінювання на test.csv.

Запуск з кореня репозиторію:
    python -m src.train_embeddings
    python -m src.train_embeddings --mode extended
    python -m src.train_embeddings --model paraphrase-multilingual-MiniLM-L12-v2
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import RANDOM_SEED
from src.embeddings import DEFAULT_MODEL, build_features, load_model

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model")
    parser.add_argument("--mode", default="concat", choices=["concat", "extended"])
    parser.add_argument("--class_weight", default=None, choices=[None, "balanced"],
                        help="'balanced'")
    args = parser.parse_args()

    #дані
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {len(train)} rows, test: {len(test)} rows")
    print("devision label у train:\n", train["label"].value_counts().sort_index())

    #Embeddings і ознаки будує тут вектори 
    model = load_model(args.model)
    X_train = build_features(train, model, args.mode)
    X_test = build_features(test, model, args.mode)
    y_train, y_test = train["label"].values, test["label"].values
    print("form X_train:", X_train.shape)

    #класифікатор навчання на лейбл
    clf = LogisticRegression(max_iter=2000, class_weight=args.class_weight, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    #оцінювання на тест
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    print(f"\naccuracy: {acc:.4f} | macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, pred, target_names=["0 irrelevant", "1 contextual", "2 core"]))
    print("confusion matrix:\n", confusion_matrix(y_test, pred))

    #сейвим
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"{args.model.replace('/', '_')}__{args.mode}"
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "mode": args.mode, "class_weight": args.class_weight,
                   "accuracy": acc, "macro_f1": macro_f1}, f, indent=2, ensure_ascii=False)
    print(f"results: results/{name}.json")


if __name__ == "__main__":
    main()