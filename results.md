# Results summary

All ranking results use the Task B development queries and rank the complete
candidate skill corpus for each job title.

## Notebooks 04-07 experiment pipeline

[`experiments/04_five_model_comparison.ipynb`](experiments/04_five_model_comparison.ipynb)
evaluates five models (MiniLM, MPNet, JobBERT, BGE, and E5) with first, mean,
and maximum skill-alias strategies. The predefined primary criterion selects
**mean aliases**: mean graded nDCG is `0.6526`, compared with `0.6502` for
first aliases and `0.6426` for maximum alias scoring.

[`experiments/05_locked_model_comparison.ipynb`](experiments/05_locked_model_comparison.ipynb)
locks the mean-alias representation. The zero-shot ranking is:

| Rank | Model | Graded nDCG | MAP | MRR |
|---:|---|---:|---:|---:|
| 1 | JobBERT | 0.7381 | 0.3416 | 0.7886 |
| 2 | E5 | 0.6401 | 0.1642 | 0.6760 |
| 3 | MPNet | 0.6327 | 0.1612 | 0.6302 |
| 4 | BGE | 0.6309 | 0.1555 | 0.6250 |
| 5 | MiniLM | 0.6213 | 0.1476 | 0.6397 |

The top three candidates, JobBERT, E5, and MPNet, were selected for
fine-tuning in
[`experiments/06_finetune_selected_models.ipynb`](experiments/06_finetune_selected_models.ipynb).

## Fine-tuning results

Fine-tuning uses `MultipleNegativesRankingLoss`, positive pairs (`label > 0`),
two epochs, batch size 128, learning rate `2e-5`, and a fixed seed.

| Model | Graded nDCG | MAP | MRR | P@1 | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|---:|
| JobBERT | 0.7319 | 0.3258 | 0.7997 | 0.6678 | 0.5882 | 0.5507 |
| E5 | 0.7040 | 0.2694 | 0.7243 | 0.5658 | 0.5138 | 0.5039 |
| MPNet | 0.6981 | 0.2602 | 0.7071 | 0.5625 | 0.5033 | 0.4766 |

## Final eight-system comparison

[`experiments/07_final_comparison_and_error_analysis.ipynb`](experiments/07_final_comparison_and_error_analysis.ipynb)
recomputed the final comparison from the complete development corpus for all
304 jobs.

| Stage | Model | nDCG graded | nDCG binary | MAP | MRR | P@1 | P@5 | P@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | JobBERT | **0.7381** | 0.7889 | **0.3416** | 0.7903 | 0.6513 | 0.5632 | 0.5178 |
| Fine-tuned | JobBERT | 0.7319 | 0.7813 | 0.3258 | **0.7997** | **0.6678** | **0.5882** | **0.5507** |
| Fine-tuned | E5 | 0.7040 | 0.7493 | 0.2694 | 0.7243 | 0.5658 | 0.5138 | 0.5039 |
| Fine-tuned | MPNet | 0.6981 | 0.7438 | 0.2602 | 0.7071 | 0.5625 | 0.5033 | 0.4766 |
| Zero-shot | E5 | 0.6401 | 0.6821 | 0.1642 | 0.6966 | 0.5428 | 0.4507 | 0.4168 |
| Zero-shot | MPNet | 0.6327 | 0.6763 | 0.1612 | 0.6300 | 0.4539 | 0.3993 | 0.3711 |
| Zero-shot | BGE | 0.6309 | 0.6720 | 0.1555 | 0.6339 | 0.4441 | 0.4132 | 0.3737 |
| Zero-shot | MiniLM | 0.6214 | 0.6645 | 0.1476 | 0.6386 | 0.4770 | 0.3836 | 0.3658 |

JobBERT zero-shot is the strongest final system by graded nDCG, MAP,
relevance AUC, and core AUC. Fine-tuned JobBERT has the strongest top-k
precision and MRR, although its nDCG and MAP are slightly lower. Fine-tuning
improves E5 graded nDCG from `0.6401` to `0.7040` and MPNet from `0.6327` to
`0.6981`.

## Error-analysis results

Notebook 07 produced 2,432 per-job rows: eight systems evaluated on each of
the 304 development jobs. It also produced the ten lowest-nDCG jobs for each
system.

| System | Relevance AUC | Core AUC |
|---|---:|---:|
| JobBERT zero-shot | **0.9596** | **0.9585** |
| JobBERT fine-tuned | 0.9420 | 0.9483 |
| E5 fine-tuned | 0.9219 | 0.9445 |
| MPNet fine-tuned | 0.9222 | 0.9436 |
| E5 zero-shot | 0.8517 | 0.8772 |
| MPNet zero-shot | 0.8543 | 0.8732 |
| BGE zero-shot | 0.8390 | 0.8744 |
| MiniLM zero-shot | 0.8303 | 0.8540 |

Generated files:

- [`results/final_system_comparison.csv`](results/final_system_comparison.csv)
- [`results/error_analysis_summary.csv`](results/error_analysis_summary.csv)
- [`results/error_analysis_per_job.csv`](results/error_analysis_per_job.csv)
- [`results/error_analysis_hardest_jobs.csv`](results/error_analysis_hardest_jobs.csv)

The hardest-job files support targeted qualitative inspection, but no manual
annotations have yet been added. Fine-tuning adapts retrieval rather than
explicitly training a separate core/contextual classifier.
