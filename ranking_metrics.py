import json
from pathlib import Path

import numpy as np


def _ranking(scores: np.ndarray) -> np.ndarray:
    # stable: однакові скори не "перемішуються" випадково між запусками
    return np.argsort(-scores, kind="stable")


def ndcg(scores: np.ndarray, grades: np.ndarray) -> float:
    """nDCG по всьому ранжуванню. grades — масив тієї ж довжини, що й scores."""
    order = _ranking(scores)
    discounts = 1.0 / np.log2(np.arange(2, len(scores) + 2))
    dcg = np.sum(grades[order] * discounts)
    ideal = np.sort(grades)[::-1]
    idcg = np.sum(ideal * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(scores: np.ndarray, grades: np.ndarray) -> float:
    """AP у бінарному режимі: core і contextual вважаються однаково релевантними."""
    order = _ranking(scores)
    rel = (grades[order] > 0).astype(float)
    n_rel = rel.sum()
    if n_rel == 0:
        return 0.0
    precision_at_k = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float(np.sum(precision_at_k * rel) / n_rel)


def reciprocal_rank(scores: np.ndarray, grades: np.ndarray) -> float:
    order = _ranking(scores)
    hits = np.flatnonzero(grades[order] > 0)
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def precision_at_k(scores: np.ndarray, grades: np.ndarray, k: int) -> float:
    order = _ranking(scores)[:k]
    return float(np.mean(grades[order] > 0))


def evaluate_query(scores: np.ndarray, grades: np.ndarray) -> dict:
    """Всі метрики для однієї посади."""
    return {
        "ndcg_graded": ndcg(scores, grades),
        "ndcg_binary": ndcg(scores, (grades > 0).astype(float)),
        "map_binary": average_precision(scores, grades),
        "mrr_binary": reciprocal_rank(scores, grades),
        "p@1": precision_at_k(scores, grades, 1),
        "p@5": precision_at_k(scores, grades, 5),
        "p@10": precision_at_k(scores, grades, 10),
    }


def mean_metrics(per_query: list) -> dict:
    keys = per_query[0].keys()
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}


def save_results(name: str, description: str, per_query: list, results_dir: str = "results") -> dict:
    """Зберігає середні метрики у results/<name>.json у спільному форматі команди:
    {"description": "...", "ndcg_graded": ..., "map_binary": ..., ...}
    name — коротка назва підходу (стане назвою файлу)."""
    metrics = mean_metrics(per_query)
    out = {"description": description, **{k: round(v, 4) for k, v in metrics.items()}}
    Path(results_dir).mkdir(exist_ok=True)
    with open(Path(results_dir) / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Збережено: {results_dir}/{name}.json")
    return out