# Job-Skill Matching, Ranking and Classification

This project investigates semantic matching between job titles and professional
skills using NLP and sentence-embedding models. It is based on **TalentCLEF
2026 Task B: Job-Skill Matching with Skill Type Classification**.

## Project objectives

The project studies the following questions:

- How well can pretrained sentence-embedding models retrieve ESCO skills for a
  job title?
- Does a job-domain model outperform general-purpose sentence-embedding models?
- Does representing a skill with all of its aliases improve retrieval?
- Does contrastive fine-tuning improve the MiniLM baseline?
- How well do the models distinguish relevant skills from irrelevant skills and
  core skills from contextual skills?

## Task B

The goal of Task B is to retrieve the skills relevant to a given job title and
classify each retrieved skill by its type:

- **Core** skills are required to perform a job independently of the work
  context or employer. They can be understood as essential skills.
- **Contextual** skills depend on factors such as the organization or industry.
  They can therefore be understood as optional skills.
- The official task description also refers to the broader **complementary** and
  **transversal** skill categories.

The task provides:

- a **training set** containing job titles and the professional skills required
  for each position;
- a **development set** containing job titles, ESCO-normalized skills, and
  core/contextual relevance annotations;
- a **test set** containing job titles for which systems must retrieve skills
  from the provided gazetteer and predict their skill type.

More information is available in the official
[Task B summary](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/#task-b-job-skill-matching-with-skill-type-classification)
and [data description](https://talentclef.github.io/talentclef/docs/talentclef-2026/data/description_corpus/).

### Dataset used in this repository

The repository contains the Task B training data and a development/validation
subset used for local evaluation:

| Data | Contents | Approximate size |
|---|---|---:|
| Raw training mappings | Job-skill relations | 114,698 rows |
| Processed training pairs | Positive pairs plus sampled negatives | 164,700 rows |
| Development queries | Job titles | 304 |
| Development corpus | Candidate skills | 9,052 |
| Development judgments | Graded qrels | 56,417 rows |

The processed training labels are:

- `0`: sampled irrelevant pair;
- `1`: contextual/optional skill;
- `2`: core/essential skill.

The 50,000 irrelevant training pairs are randomly sampled job-skill
combinations that do not occur in the provided positive mapping. They are
assumed negatives for the supervised logistic-regression baseline; they are not
independently annotated negatives. The fine-tuning experiment uses only
positive pairs (`label > 0`).

`data/processed/test.csv` is a locally generated, validation-derived table
with sampled negatives. It is **not** the official hidden Task B test set. The
main ranking evaluation instead uses the raw development queries, complete
skill corpus, and `qrels.tsv`.

## Repository contents

### Data

- `data/raw/TaskB/training/` contains the ESCO training mappings and aliases.
- `data/raw/TaskB/validation/` contains development queries, the skill
  corpus, and graded relevance judgments.
- `data/processed/train.csv` contains job/skill text pairs with labels:
  `0` = irrelevant, `1` = contextual, and `2` = core.
- `data/processed/test.csv` is a processed validation-derived table with
  sampled irrelevant pairs.

The data-processing workflow is documented in
[`data_processing.ipynb`](data_processing.ipynb).

### Models and experiments

- [`baseline.ipynb`](baseline.ipynb) evaluates the
  `all-MiniLM-L6-v2` zero-shot cosine baseline and logistic-regression
  alternatives.
- [`experiments/01_models_and_aliases.ipynb`](experiments/01_models_and_aliases.ipynb)
  compares MiniLM, MPNet, and JobBERT, including different skill-alias
  aggregation strategies.
- [`experiments/02_finetune_minilm.ipynb`](experiments/02_finetune_minilm.ipynb)
  fine-tunes MiniLM with contrastive learning on positive job-skill pairs.
- [`experiments/03_error_analysis.ipynb`](experiments/03_error_analysis.ipynb)
  compares models per job and prepares false positives for analysis.

Ranking metrics are implemented in
[`ranking_metrics.py`](ranking_metrics.py). Saved experiment results are in
[`results/`](results/).

## Methodology

The common evaluation pipeline is:

```text
Task B / ESCO data
        ↓
Select job and skill text, including alias strategy
        ↓
Encode job titles and candidate skills
        ↓
Compute cosine similarity or learned pair scores
        ↓
Rank the complete development skill corpus
        ↓
Compare rankings with graded qrels
```

The baseline uses `all-MiniLM-L6-v2` embeddings and cosine similarity. Two
logistic-regression variants use the same embeddings as pair features. The
model comparison evaluates MiniLM, MPNet, and JobBERT. The fine-tuning
experiment adapts MiniLM with `MultipleNegativesRankingLoss` using positive
job-skill pairs.

The fine-tuning experiment improves retrieval relevance, but it does not train
an explicit core/contextual classifier: labels `1` and `2` are both treated as
positive matches. The distinction is evaluated through graded metrics.

## Evaluation

The development evaluation ranks the full skill corpus for each job title and
uses graded relevance judgments. The primary metrics used in this project are:

- **nDCG**: graded nDCG distinguishes core from contextual relevance.
- **MAP**: binary average precision treats both core and contextual skills as
  relevant.
- **MRR**: measures the rank of the first relevant skill.
- **Precision@1, Precision@5, and Precision@10**.

The official task also identifies nDCG and MAP as information-retrieval
evaluation metrics and reports MRR and Precision@K (K = 1, 5, 10).
Further details are available on the official
[evaluation page](https://talentclef.github.io/talentclef/docs/talentclef-2026/evaluation/).

For binary metrics, both core and contextual skills are considered relevant.
For graded nDCG, core skills have grade `2` and contextual skills have grade
`1`. All main ranking experiments evaluate every development job against the
full 9,052-skill corpus rather than only against the sampled negatives in the
processed CSV files.

## Current results

The current experiments include the following approaches:

- zero-shot cosine similarity with MiniLM, MPNet, and JobBERT;
- first, mean, and maximum aggregation of skill aliases;
- logistic regression over embedding-based pair features;
- contrastive fine-tuning of MiniLM with
  `MultipleNegativesRankingLoss`.

The best stored graded nDCG is currently **0.7381**, achieved by
JobBERT with mean skill-alias embeddings. The fine-tuned MiniLM model reaches
**0.6744** graded nDCG, improving over the original MiniLM zero-shot baseline
at **0.6154**.

See [`results.md`](results.md) for the complete current results table and a
short interpretation.

## Limitations

- Sampled negative pairs may contain valid but unrecorded job-skill relations.
- The fine-tuned model learns relevance but does not explicitly classify core
  versus contextual skills.
- Some experiments use only the first skill alias; alias aggregation is not
  applied uniformly to every experiment.
- The trained `models/minilm_mnrl` weights are not stored in the repository.
- Pretrained model downloads require access to Hugging Face.
- Dependency versions are not pinned in `requirements.txt`.
- Results report means over development queries but do not include confidence
  intervals or significance tests.
- The error-analysis CSV is prepared for annotation, but its error-type and
  note fields are not yet completed.

## Project report

[`report.md`](report.md) provides the project-oriented description of the
problem, data, methods, experiments, results, limitations, and conclusions.

## Installation

Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

The notebooks download pretrained models from Hugging Face on first use.
Run notebooks from the repository root unless a notebook explicitly changes
its working directory.
