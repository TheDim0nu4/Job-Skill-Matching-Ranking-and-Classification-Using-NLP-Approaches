# Job-Skill Matching with Skill Type Classification

## Abstract

This project investigates retrieval of professional skills for job titles in
the TalentCLEF 2026 Task B setting. Sentence embeddings are used to rank a
complete ESCO-normalized skill corpus for each development job title. The
experiments compare general-purpose models, a job-domain model, skill-alias
aggregation strategies, logistic-regression scoring, and contrastive
fine-tuning of MiniLM.

The strongest stored result is obtained by `TechWolf/JobBERT-v2` with mean
skill-alias embeddings, achieving graded nDCG of `0.7381`. Fine-tuning
`all-MiniLM-L6-v2` improves its graded nDCG from `0.6154` to `0.6744`, but
does not outperform JobBERT. The analysis also shows that relevance retrieval
is substantially stronger than core-versus-contextual ordering.

## 1. Problem and objectives

The task is to retrieve skills relevant to a job title and assign relevance
according to the Task B annotations. Core skills are required to perform the
job independently of the employer or work context. Contextual skills depend
more on the organization or industry and are optional in that sense.

The project addresses these questions:

1. How effective are pretrained sentence embeddings for job-skill ranking?
2. Does domain specialization improve over general-purpose models?
3. Does using skill aliases improve retrieval?
4. Does contrastive fine-tuning improve MiniLM?
5. Are models better at retrieving relevant skills than distinguishing core
   from contextual skills?

## 2. Data

The raw data is organized according to Task B:

- training job-skill mappings and job/skill aliases are in
  `data/raw/TaskB/training/`;
- development queries, candidate skills, and graded qrels are in
  `data/raw/TaskB/validation/`.

The development evaluation contains 304 job titles and 9,052 candidate skills.
The qrels use grade `2` for core skills and grade `1` for contextual skills.
The processed training data contains approximately 164,700 pairs: the known
positive relations plus 50,000 sampled negative pairs labelled `0`.

The sampled negatives are not independently annotated. They are used for the
supervised logistic-regression baseline. The fine-tuning experiment filters
the processed data to positive pairs and therefore treats core and contextual
skills alike during training.

## 3. Methods

### 3.1 Embedding ranking

Each job title and candidate skill is encoded as a sentence embedding. Because
the embeddings are normalized, cosine similarity is calculated as a dot
product. Skills are ranked by decreasing similarity for every job.

The compared pretrained models are:

- `all-MiniLM-L6-v2`, a compact general-purpose sentence-embedding model;
- `all-mpnet-base-v2`, a larger general-purpose embedding model;
- `TechWolf/JobBERT-v2`, a model specialized for job-related text.

### 3.2 Alias strategies

The ESCO skill corpus provides alternative names for skills. The experiments
compare:

- **first**: use only the first alias;
- **mean**: average the embeddings of all aliases;
- **max**: score every alias and use the highest similarity.

Job queries use the available job-title text; the alias variation therefore
primarily affects skill representations.

### 3.3 Logistic-regression baselines

Two supervised models use MiniLM embeddings as pair features:

- concatenation of the job and skill embeddings;
- an extended representation containing concatenation, absolute difference,
  elementwise product, and cosine similarity.

The classifier predicts labels `0`, `1`, and `2`. Its ranking score is
`P(label=1) + 2 * P(label=2)`.

### 3.4 Contrastive fine-tuning

MiniLM is fine-tuned with `MultipleNegativesRankingLoss`. A relevant skill is
paired with a job title, while other skills in the batch act as negatives.
Training uses two epochs, batch size 128, learning rate `2e-5`, a fixed seed,
and a job-level validation split. Since all rows with `label > 0` are treated
as positives, this experiment adapts retrieval but does not explicitly train
core/contextual classification.

## 4. Evaluation

The main evaluation ranks the full development skill corpus for each query.
It does not evaluate only the artificial negatives in
`data/processed/test.csv`.

The reported metrics are:

- graded nDCG, where core has greater gain than contextual relevance;
- binary nDCG and MAP, where both relevant categories are positive;
- MRR;
- Precision@1, Precision@5, and Precision@10.

The implementation is in `ranking_metrics.py`. Results are averaged across
development queries.

## 5. Results

The complete current result table is in [`results.md`](results.md).
The main findings are:

- JobBERT with mean aliases has the best graded nDCG: `0.7381`.
- JobBERT with the first alias has the best MAP (`0.3479`), MRR (`0.8210`),
  and P@1 (`0.6974`).
- Fine-tuned MiniLM improves over zero-shot MiniLM from `0.6154` to `0.6744`
  graded nDCG.
- MPNet improves over MiniLM but remains below JobBERT.
- Maximum alias scoring is weaker than first or mean alias strategies for all
  three embedding models.
- Both logistic-regression variants are weaker than direct cosine ranking.

## 6. Error analysis

The error-analysis notebook compares JobBERT and fine-tuned MiniLM for each of
the 304 development jobs. JobBERT has the higher per-job graded nDCG for 263
jobs, while fine-tuned MiniLM is higher for 41 jobs.

JobBERT separates relevant from irrelevant skills well, with an approximate
pair AUC of `0.959`; fine-tuned MiniLM reaches approximately `0.904`.
Core-versus-contextual ordering is much weaker, with approximate AUC values of
`0.544` and `0.602`, respectively. This indicates that retrieving relevant
skills is easier than ranking core skills above contextual skills.

The notebook generates per-job tables and a sample of 100 false positives.
The false-positive file is prepared for manual annotation, but its taxonomy
and annotation fields are not yet complete.

## 7. Limitations

- Sampled negatives may include valid relations that are absent from the
  provided mapping.
- Fine-tuning does not explicitly classify core versus contextual skills.
- The experiments do not report confidence intervals or significance tests.
- The trained MiniLM weights are not included in the repository.
- Model downloads depend on external Hugging Face availability.
- Package versions are not pinned.
- Alias handling is not identical across all experiments.
- The manual false-positive analysis is preliminary.
- Development results should not be interpreted as official hidden-test
  performance.

## 8. Conclusion

The experiments show that semantic embeddings provide a useful baseline for
job-skill retrieval. Domain specialization is important: JobBERT substantially
outperforms the general-purpose models. Contrastive fine-tuning is beneficial
for MiniLM, but the available training setup does not close the gap to the
job-domain model. The main remaining technical challenge is reliable
core-versus-contextual classification, which is only weakly captured by the
current similarity-based ranking approach.

## References

- [TalentCLEF 2026 Task B summary](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/#task-b-job-skill-matching-with-skill-type-classification)
- [TalentCLEF data description](https://talentclef.github.io/talentclef/docs/talentclef-2026/data/description_corpus/)
- [TalentCLEF evaluation page](https://talentclef.github.io/talentclef/docs/talentclef-2026/evaluation/)
