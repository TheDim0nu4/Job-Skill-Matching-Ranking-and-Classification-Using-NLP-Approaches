# Phase 2 Experiment Plan

## Objective

Extend the Phase 1 JobBERT retrieval baseline with explicit skill-type
classification and reranking. The final pipeline should retrieve candidate
skills, classify each candidate as non-relevant, contextual, or core, and use
that information to produce a better final ranking.

The saved Phase 1 results remain the baseline. They should not be retrained or
recomputed unless a reproducibility check is required.

## Phase 1 baseline

Use the existing results as the fixed reference:

- zero-shot JobBERT with mean aliases;
- fine-tuned JobBERT with mean aliases;
- the full development corpus and qrels;
- graded nDCG, MAP, MRR, and Precision@K.

The authoritative aggregate results are stored in
`results/final_system_comparison.csv` and
`results/finetuning_selected_models.csv`.

## Experiment 1: Candidate recall

Measure whether JobBERT retrieves relevant skills into a restricted candidate
pool before adding a classifier. Evaluate top-10, top-25, top-50, top-100,
top-200, top-500, top-1000, top-2000, and top-5000 candidates.

Report:

- Recall@K for all relevant skills;
- Recall@K for core skills;
- the computational cost of each candidate-pool size.

Select K using both recall and inference cost. The first run showed that
top-50 and top-100 are not safe pools: they capture only about 12--14% and
23--24% of all relevant skills, respectively. The measured micro Recall@1000
was 92.1% for zero-shot JobBERT and 81.9% for fine-tuned JobBERT; at top-2000
it was 96.9% and 93.6%. These results should guide the later classifier
experiment rather than assuming that top-50 or top-100 is sufficient.

The implementation and outputs are:

- `experiments/phase2_01_candidate_recall.py`;
- `results/phase2_candidate_recall_summary.csv`;
- `results/phase2_candidate_recall_per_query.csv`.

This prevents the classifier from being evaluated on a candidate pool that
already omits many correct skills.

## Experiment 2: MLP embedding classifier

Train a small multilayer perceptron (MLP) using existing JobBERT
representations. The primary input should combine the job embedding, skill
embedding, their element-wise interaction features, and cosine similarity.
The MLP should have a small hidden layer and a three-class output; it is not
intended to replace JobBERT or become a new language model.

An optional logistic-regression classifier can be used as a simple sanity
baseline, but the MLP is the main Experiment 2 model. Its output is:

- `0`: non-relevant;
- `1`: contextual;
- `2`: core.

Use the processed training pairs, mapping `essential` to `2`, `optional` to
`1`, and synthetic negatives to `0`. Split by job title rather than by
individual rows. Select the hidden size, dropout, learning rate, and training
epochs using the training-derived validation split.

Evaluate macro-F1, per-class precision and recall, balanced accuracy, and a
confusion matrix.

## Experiment 3: Cross-encoder classifier

Train a transformer sequence-classification model that receives the job title
and skill text together as one pair, for example:

```text
job title [SEP] skill text
```

The model has a three-class classification head:

- `0`: non-relevant;
- `1`: contextual;
- `2`: core.

Initialize it from a pretrained transformer suitable for English sentence
pairs. If the JobBERT backbone can be used reliably with a sequence
classification head, it is preferable for domain consistency; otherwise use a
documented pretrained cross-encoder backbone and fine-tune it on the project
labels. This model is not run against all 9,052 skills. Apply it only to the
top-K candidates selected by Experiment 1.

Train it with cross-entropy on the same job-level splits and labels as the MLP
classifier. Select training hyperparameters on the training-derived
validation split. Compare it directly with the MLP using the same candidate
pairs, labels, and classification metrics. The cross-encoder should model
job-skill interactions more accurately, while the MLP should be faster at
inference.

## Experiment 4: Reranking ablation

Compare the following systems:

1. JobBERT retrieval only;
2. JobBERT retrieval followed by classifier filtering;
3. JobBERT retrieval with classifier probabilities used for reranking;
4. JobBERT retrieval followed by cross-encoder classification and reranking.

The reranker should combine the initial retrieval score with the predicted
skill type. Any weighting parameters must be selected on training-derived
validation data, not tuned on the final development benchmark.

## Experiment 5: Final comparison

Compare the best Phase 2 system with the fixed Phase 1 systems:

- zero-shot JobBERT;
- fine-tuned JobBERT;
- JobBERT plus embedding classifier;
- JobBERT plus cross-encoder classifier;
- JobBERT plus classifier and reranking.

Report retrieval metrics (graded nDCG, MAP, MRR, and Precision@K) separately
from classification metrics (macro-F1, per-class precision and recall,
balanced accuracy, and confusion matrix). Explain trade-offs when a system
improves classification but does not improve every ranking metric.

## Experiment 6: Error analysis

Categorize errors into:

- retrieval failure: a relevant skill was not present in top-K;
- classification failure: a candidate received the wrong class;
- reranking failure: the class was predicted correctly but the final order
  was still poor.

Inspect the hardest queries, quantify the frequency of each error category,
and derive at least one targeted improvement. Validate that improvement on a
separate validation split before reporting it as a result.

## Recommended execution order

```text
Fixed Phase 1 baseline
        ↓
Candidate Recall@K
        ↓
Embedding classifier
        ↓
Cross-encoder classifier
        ↓
Reranking ablation
        ↓
Final comparison
        ↓
Error analysis
```

The main Phase 2 comparison is:

```text
fine-tuned JobBERT
vs.
fine-tuned JobBERT + cross-encoder classifier + reranking
```

This keeps the successful Phase 1 retrieval component and adds the explicit
skill-type prediction that is not implemented in the current pipeline.
