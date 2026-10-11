import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Англійська модель: швидка, добрий старт.
# Якщо в даних є іспанська/німецька/китайська — візьми:
# "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(texts, model: SentenceTransformer, batch_size: int = 64) -> dict:
    #кодує унікальні тексти за для того щоб це було швидше 
    unique = sorted(set(texts))
    vectors = model.encode(
        unique,
        batch_size=batch_size,
        normalize_embeddings=True,  #довжина вектора 1 тоді cosine=скалярний добуток
        show_progress_bar=True,
    )
    return dict(zip(unique, vectors))


def build_features(df: pd.DataFrame, model: SentenceTransformer, mode: str = "concat") -> np.ndarray:
    jobs = df["job_name"].astype(str).tolist()
    skills = df["skill_name"].astype(str).tolist()

    job_map = encode_texts(jobs, model)
    skill_map = encode_texts(skills, model)

    E_job = np.vstack([job_map[j] for j in jobs])
    E_skill = np.vstack([skill_map[s] for s in skills])

    if mode == "concat":
        return np.hstack([E_job, E_skill])

    if mode == "extended":
        cosine = np.sum(E_job * E_skill, axis=1, keepdims=True)
        return np.hstack([E_job, E_skill, np.abs(E_job - E_skill), E_job * E_skill, cosine])

    raise ValueError(f"Невідомий mode: {mode}. Використай 'concat' або 'extended'.")