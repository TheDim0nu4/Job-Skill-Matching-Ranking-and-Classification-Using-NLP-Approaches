# Job-Skill Matching with Skill Type Classification

## Abstract

This project investigates retrieval of professional skills for job titles in
the TalentCLEF 2026 Task B setting. Sentence embeddings are used to rank a
complete ESCO-normalized skill corpus for each development job title. The
experiments compare general-purpose models, a job-domain model, skill-alias
aggregation strategies, and contrastive fine-tuning of three selected models.

The strongest stored result is obtained by `TechWolf/JobBERT-v2` with mean
skill-alias embeddings, achieving graded nDCG of `0.7381`. Fine-tuned JobBERT, E5, and MPNet reach graded nDCG values of `0.7319`,
`0.7040`, and `0.6981`. The final analysis shows that relevance retrieval is
substantially stronger than core-versus-contextual ordering, while JobBERT
zero-shot remains the strongest overall system.

## 1. Problem and objectives

The task is to retrieve skills relevant to a job title and assign relevance
according to the Task B annotations. Core skills are required to perform the
job independently of the employer or work context. Contextual skills depend
more on the organization or industry and are optional in that sense.

The project addresses these questions:

1. How effective are pretrained sentence embeddings for job-skill ranking?
2. Does domain specialization improve over general-purpose models?
3. Does using skill aliases improve retrieval?
4. Does contrastive fine-tuning improve the selected models?
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

The sampled negatives are not independently annotated. The fine-tuning
experiment filters the processed data to positive pairs and therefore treats
core and contextual skills alike during training.

## 3. Methods

### 3.1 Embedding ranking

Each job title and candidate skill is encoded as a sentence embedding. Because
the embeddings are normalized, cosine similarity is calculated as a dot
product. Skills are ranked by decreasing similarity for every job.

The compared pretrained models are:

- `all-MiniLM-L6-v2`, a compact general-purpose sentence-embedding model;
- `all-mpnet-base-v2`, a larger general-purpose embedding model;
- `TechWolf/JobBERT-v2`, a model specialized for job-related text.
- `BAAI/bge-base-en-v1.5`, a retrieval-oriented embedding model;
- `intfloat/e5-base-v2`, an embedding model with query/passage prompting.

### 3.2 Alias strategies

The ESCO skill corpus provides alternative names for skills. The experiments
compare:

- **first**: use only the first alias;
- **mean**: average the embeddings of all aliases;
- **max**: score every alias and use the highest similarity.

Job queries use the available job-title text; the alias variation therefore
primarily affects skill representations.

### 3.3 Contrastive fine-tuning

JobBERT, E5, and MPNet are fine-tuned with `MultipleNegativesRankingLoss`.
A relevant skill is paired with a job title, while other skills in the batch
act as negatives. Training uses two epochs, batch size 128, learning rate
`2e-5`, a fixed seed, and a job-level validation split. Since all rows with
`label > 0` are treated as positives, this experiment adapts retrieval but
does not explicitly train core/contextual classification.

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
- JobBERT zero-shot with mean aliases has the best final graded nDCG (`0.7381`).
- Fine-tuned JobBERT has the best final MRR (`0.7997`) and P@1/P@5/P@10.
- Fine-tuning improves E5 from `0.6401` to `0.7040` graded nDCG and MPNet
  from `0.6327` to `0.6981`.

## 6. Error analysis

The final error-analysis notebook evaluates eight systems for each of the 304
development jobs, producing 2,432 per-job rows and the ten hardest jobs for
each system. JobBERT zero-shot achieves the highest aggregate relevance AUC
(`0.9596`) and core AUC (`0.9585`). Fine-tuned JobBERT reaches `0.9420`
relevance AUC and `0.9483` core AUC. Fine-tuned E5 and MPNet reach relevance
AUC values of `0.9219` and `0.9222`.

The hardest-job files support targeted qualitative inspection, but no manual
annotation has yet been completed.

## 7. Conclusion

The experiments show that semantic embeddings provide a useful baseline for
job-skill retrieval. Domain specialization is important: JobBERT substantially
outperforms the general-purpose models. Contrastive fine-tuning improves E5
and MPNet considerably and improves top-k precision for JobBERT, but JobBERT
zero-shot remains best by overall graded nDCG. The main remaining technical
challenge is reliable core-versus-contextual classification, which is only
weakly captured by the current similarity-based ranking approach.

## References

- [TalentCLEF 2026 Task B summary](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/#task-b-job-skill-matching-with-skill-type-classification)
- [TalentCLEF data description](https://talentclef.github.io/talentclef/docs/talentclef-2026/data/description_corpus/)
- [TalentCLEF evaluation page](https://talentclef.github.io/talentclef/docs/talentclef-2026/evaluation/)
