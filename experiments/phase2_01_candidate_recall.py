"""Measure Phase 1 JobBERT candidate recall for Phase 2 candidate pools."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "raw" / "TaskB" / "validation"
RESULTS_DIR = ROOT / "results"
TOP_KS = (10, 25, 50, 100, 200, 500, 1000, 2000, 5000)
BATCH_SIZE = 256
MODEL_CONFIGS = {
    "jobbert_zero_shot": "TechWolf/JobBERT-v2",
    "jobbert_fine_tuned": str(ROOT / "models" / "jobbert_mnrl_mean_alias"),
}


def load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    queries = pd.read_csv(VAL_DIR / "queries", sep="\t")
    corpus = pd.read_csv(VAL_DIR / "corpus_elements", sep="\t")
    qrels = pd.read_csv(
        VAL_DIR / "qrels.tsv",
        sep="\t",
        header=None,
        names=["q_id", "iteration", "c_id", "relevance"],
    )

    corpus["skill_aliases"] = corpus["skill_aliases"].map(ast.literal_eval)
    corpus["skill_aliases"] = corpus["skill_aliases"].map(
        lambda aliases: [str(alias).strip() for alias in aliases if str(alias).strip()]
    )
    if corpus["skill_aliases"].map(len).eq(0).any():
        raise ValueError("The validation corpus contains a skill without aliases")

    skill_ids = corpus["c_id"].tolist()
    skill_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    grades_by_query: dict[str, np.ndarray] = {}
    for query_id, query_qrels in qrels.groupby("q_id", sort=False):
        grades = np.zeros(len(skill_ids), dtype=np.float32)
        for row in query_qrels.itertuples(index=False):
            if row.c_id in skill_index:
                grades[skill_index[row.c_id]] = row.relevance
        grades_by_query[str(query_id)] = grades

    queries["q_id"] = queries["q_id"].astype(str)
    grades_by_query = {
        str(query_id): grades
        for query_id, grades in grades_by_query.items()
        if str(query_id) in set(queries["q_id"])
    }
    if not grades_by_query:
        raise ValueError("No development queries have qrels")

    return queries, corpus, grades_by_query


def encode_mean_aliases(
    model: SentenceTransformer,
    aliases_by_skill: list[list[str]],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    counts = [len(aliases) for aliases in aliases_by_skill]
    flattened = [alias for aliases in aliases_by_skill for alias in aliases]
    encoded = model.encode(
        flattened,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    skill_embeddings = []
    start = 0
    for count in counts:
        mean_embedding = encoded[start : start + count].mean(axis=0)
        mean_embedding /= max(np.linalg.norm(mean_embedding), 1e-12)
        skill_embeddings.append(mean_embedding)
        start += count
    return np.asarray(skill_embeddings, dtype=np.float32)


def recall_rows(
    model_name: str,
    scores: np.ndarray,
    query_ids: list[str],
    grades_by_query: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order = np.argsort(-scores, axis=1, kind="stable")
    for query_index, query_id in enumerate(query_ids):
        grades = grades_by_query[query_id]
        relevant = grades > 0
        core = grades == 2
        contextual = grades == 1
        ranking = order[query_index]
        for k in TOP_KS:
            selected = ranking[:k]
            row = {
                "model": model_name,
                "q_id": query_id,
                "k": k,
                "n_relevant": int(relevant.sum()),
                "n_core": int(core.sum()),
                "n_contextual": int(contextual.sum()),
                "relevant_found": int(relevant[selected].sum()),
                "core_found": int(core[selected].sum()),
                "contextual_found": int(contextual[selected].sum()),
                "relevant_hit": bool(relevant[selected].any()),
                "core_hit": bool(core[selected].any()),
            }
            row["relevant_recall"] = row["relevant_found"] / row["n_relevant"]
            row["core_recall"] = (
                row["core_found"] / row["n_core"] if row["n_core"] else np.nan
            )
            row["contextual_recall"] = (
                row["contextual_found"] / row["n_contextual"]
                if row["n_contextual"]
                else np.nan
            )
            rows.append(row)
    return rows


def main() -> int:
    queries, corpus, grades_by_query = load_evaluation_data()
    query_ids = [query_id for query_id in queries["q_id"] if query_id in grades_by_query]
    query_texts = queries.set_index("q_id").loc[query_ids, "jobtitle"].tolist()
    aliases_by_skill = corpus["skill_aliases"].tolist()

    all_rows: list[dict[str, object]] = []
    for model_name, model_path in MODEL_CONFIGS.items():
        print(f"Loading {model_name}: {model_path}")
        model = SentenceTransformer(model_path)
        query_embeddings = model.encode(
            query_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        skill_embeddings = encode_mean_aliases(model, aliases_by_skill)
        scores = query_embeddings @ skill_embeddings.T
        all_rows.extend(recall_rows(model_name, scores, query_ids, grades_by_query))

    per_query = pd.DataFrame(all_rows)
    summary = (
        per_query.groupby(["model", "k"], as_index=False)
        .agg(
            n_queries=("q_id", "nunique"),
            n_candidate_skills=("n_relevant", lambda _: len(corpus)),
            n_relevant=("n_relevant", "sum"),
            n_core=("n_core", "sum"),
            n_contextual=("n_contextual", "sum"),
            relevant_found=("relevant_found", "sum"),
            core_found=("core_found", "sum"),
            contextual_found=("contextual_found", "sum"),
            mean_relevant_recall=("relevant_recall", "mean"),
            mean_core_recall=("core_recall", "mean"),
            mean_contextual_recall=("contextual_recall", "mean"),
            query_coverage=("relevant_hit", "mean"),
            core_query_coverage=("core_hit", "mean"),
        )
        .sort_values(["model", "k"])
    )
    summary["micro_relevant_recall"] = summary["relevant_found"] / summary["n_relevant"]
    summary["micro_core_recall"] = summary["core_found"] / summary["n_core"]
    summary["micro_contextual_recall"] = (
        summary["contextual_found"] / summary["n_contextual"]
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    per_query.to_csv(RESULTS_DIR / "phase2_candidate_recall_per_query.csv", index=False)
    summary.to_csv(RESULTS_DIR / "phase2_candidate_recall_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Saved Phase 2 candidate-recall outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
