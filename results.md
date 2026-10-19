# Results summary

This file summarizes the results currently stored in `results/*.json`.
All ranking results are calculated on the Task B development queries by
ranking the complete candidate skill corpus for each job title.

## Main comparison

| Approach | Model | Skill representation | nDCG graded | nDCG binary | MAP | MRR | P@1 | P@5 | P@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot cosine | JobBERT | Mean of aliases | **0.7381** | 0.7889 | 0.3416 | 0.7886 | 0.6480 | 0.5632 | 0.5178 |
| Zero-shot cosine | JobBERT | First alias | 0.7359 | **0.7899** | **0.3479** | **0.8210** | **0.6974** | **0.5645** | 0.4914 |
| Zero-shot cosine | JobBERT | Maximum alias score | 0.7223 | 0.7723 | 0.3077 | 0.7535 | 0.5987 | 0.5125 | 0.4816 |
| Fine-tuned cosine | MiniLM | First alias | 0.6744 | 0.7217 | 0.2273 | 0.6643 | 0.5000 | 0.4704 | 0.4441 |
| Zero-shot cosine | MPNet | Mean of aliases | 0.6327 | 0.6763 | 0.1612 | 0.6302 | 0.4539 | 0.3980 | 0.3711 |
| Zero-shot cosine | MPNet | First alias | 0.6251 | 0.6704 | 0.1545 | 0.6135 | 0.4408 | 0.3809 | 0.3549 |
| Zero-shot cosine | MiniLM | Mean of aliases | 0.6214 | 0.6645 | 0.1477 | 0.6397 | 0.4803 | 0.3836 | 0.3658 |
| Zero-shot cosine | MiniLM | First alias | 0.6154 | 0.6599 | 0.1430 | 0.6206 | 0.4342 | 0.3829 | 0.3539 |
| Zero-shot cosine | MPNet | Maximum alias score | 0.6189 | 0.6625 | 0.1446 | 0.6059 | 0.4178 | 0.3783 | 0.3523 |
| Zero-shot cosine | MiniLM | Maximum alias score | 0.6105 | 0.6536 | 0.1360 | 0.5901 | 0.3882 | 0.3730 | 0.3464 |
| Logistic regression | MiniLM | Concatenated embeddings | 0.4856 | 0.5213 | 0.0371 | 0.0947 | 0.0099 | 0.0303 | 0.0461 |
| Logistic regression | MiniLM | Extended pair features | 0.5902 | 0.6262 | 0.1137 | 0.3011 | 0.1151 | 0.1526 | 0.1842 |

## Short interpretation

- JobBERT is the strongest overall model in the stored experiments.
- JobBERT with mean alias embeddings has the best graded nDCG (`0.7381`).
- JobBERT with the first alias has the best MAP, MRR, and P@1.
- Mean alias aggregation improves MiniLM and MPNet over their first-alias
  variants, but maximum alias scoring is consistently weaker.
- Fine-tuning MiniLM improves graded nDCG from `0.6154` to `0.6744`.
- Direct cosine ranking is stronger than both logistic-regression variants.
- The results measure retrieval and graded ranking. Fine-tuning does not
  explicitly train a separate core/contextual classifier.

## Error-analysis results

The error-analysis notebook compares JobBERT with fine-tuned MiniLM for 304
development jobs:

- JobBERT has higher per-job nDCG for 263 jobs.
- Fine-tuned MiniLM has higher per-job nDCG for 41 jobs.
- JobBERT mean per-job graded nDCG is approximately `0.736`.
- Fine-tuned MiniLM mean per-job graded nDCG is approximately `0.674`.
- JobBERT relevant-versus-irrelevant pair AUC is approximately `0.959`.
- Fine-tuned MiniLM relevant-versus-irrelevant pair AUC is approximately
  `0.904`.
- Core-versus-contextual pair AUC is weak for both models: approximately
  `0.544` for JobBERT and `0.602` for fine-tuned MiniLM.

The repository also contains a 100-row false-positive sample in
`results/error_analysis/false_positives_to_annotate.csv`. Its annotation
columns are currently empty, so it represents preliminary rather than
completed manual error analysis.
