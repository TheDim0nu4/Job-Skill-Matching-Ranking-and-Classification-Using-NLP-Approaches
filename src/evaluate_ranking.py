import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import RANDOM_SEED
from src.embeddings import DEFAULT_MODEL, encode_texts, load_model, make_features
from src.ranking_metrics import evaluate_query, mean_metrics

VAL_DIR = Path("data/raw/TaskB/validation")
TRAIN_PATH = Path("data/processed/train.csv")
RESULTS_DIR = Path("results")


def load_validation():
    queries = pd.read_csv(VAL_DIR / "queries", sep="\t").rename(columns={"jobtitle": "job_name", "q_id": "job_id"})
    corpus = pd.read_csv(VAL_DIR / "corpus_elements", sep="\t").rename(columns={"c_id": "skill_id"})
    corpus["skill_aliases"] = corpus["skill_aliases"].apply(ast.literal_eval)
    corpus["skill_name"] = corpus["skill_aliases"].apply(lambda x: x[0])  # як у data_processing.ipynb
    qrels = pd.read_csv(VAL_DIR / "qrels.tsv", header=None, sep="\t", names=["job_id", "_", "skill_id", "label"])
    return queries, corpus, qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scorer", default="cosine", choices=["cosine", "lr"])
    parser.add_argument("--mode", default="extended", choices=["concat", "extended"],
                        help="ознаки для --scorer lr")
    args = parser.parse_args()

    queries, corpus, qrels = load_validation()
    print(f"queries: {len(queries)}, corpus: {len(corpus)}, qrels: {len(qrels)}")

    model = load_model(args.model)
    job_map = encode_texts(queries["job_name"].tolist(), model)
    skill_map = encode_texts(corpus["skill_name"].tolist(), model)
    skill_emb = np.vstack([skill_map[s] for s in corpus["skill_name"]])

    # (job_id, skill_id) -> grade; все інше = 0
    skill_index = {sid: i for i, sid in enumerate(corpus["skill_id"])}
    grades_by_job = {}
    for r in qrels.itertuples(index=False):
        g = grades_by_job.setdefault(r.job_id, np.zeros(len(corpus)))
        g[skill_index[r.skill_id]] = r.label

    clf = None
    if args.scorer == "lr":
        train = pd.read_csv(TRAIN_PATH)
        tj = encode_texts(train["job_name"].astype(str).tolist(), model)
        ts = encode_texts(train["skill_name"].astype(str).tolist(), model)
        X = make_features(np.vstack([tj[j] for j in train["job_name"]]),
                          np.vstack([ts[s] for s in train["skill_name"]]), args.mode)
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED).fit(X, train["label"].values)

    per_query = []
    for row in queries.itertuples(index=False):
        if row.job_id not in grades_by_job:
            continue  # посада без оцінок (в офіційному скрипті такі теж пропускаються)
        e_job = job_map[row.job_name]
        if args.scorer == "cosine":
            scores = skill_emb @ e_job  # вектори нормалізовані це cosine
        else:
            feats = make_features(np.tile(e_job, (len(corpus), 1)), skill_emb, args.mode)
            proba = clf.predict_proba(feats)  # колонки: класи 0, 1, 2
            scores = proba[:, 1] + 2 * proba[:, 2]  # очікувана релевантність
        per_query.append(evaluate_query(scores, grades_by_job[row.job_id]))

    result = mean_metrics(per_query)
    print(f"\nОцінено посад: {len(per_query)}")
    for k, v in result.items():
        print(f"{k:12s} {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"ranking__{args.model.replace('/', '_')}__{args.scorer}" + (f"_{args.mode}" if clf else "")
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "scorer": args.scorer, "mode": args.mode if clf else None,
                   "n_queries": len(per_query), **result}, f, indent=2)
    print(f"Збережено: results/{name}.json")


if __name__ == "__main__":
    main()