# Use of LLMs and AI Agents — [Anatolii Vasylov], embeddings part

This appendix documents all use of large language models in my part of the project

## Tool

- **Assistant:** Claude (Anthropic), via the claude.ai chat interface
- **No autonomous agent** modified the repository. I ran all the code myself on my own computer (MacBook, Apple Silicon)
  and committed it myself.

## General role of the LLM

The LLM wrote the first versions of the code for my part and the analysis scripts, helped interpret
the results, and reviewed the README and the team report.
I ran all the experiments myself, fed the real outputs and errors back into the chat, and decided myself
which direction to take.

## Log by stage

| # | What I asked (summary of my prompts) | What the LLM produced | What I did / verified |
|---|---|---|---|
| 1 | Sent the task plan from my teammate; asked for help implementing the embeddings part | A plan; first code: `embeddings.py`, `requirements.txt` (pretrained `all-MiniLM-L6-v2`) | Cloned the repository, created my branch, ran both experiments, reported the outputs |
| 2 | Sent the course requirements and the TalentCLEF Task B page; | Read the official evaluation pages and the Task B overview paper; stated that the official metric is ranking (graded nDCG, MAP), not F1; | Ran the evaluation; saw that the cosine baseline is better than LogisticRegression; |
| 3 | Sent `data_processing.ipynb` and a data sample; asked | Explained how the data was built; found that negative generation via `set` without `sorted` is not reproducible between runs; suggested `sorted(...)` | Applied the fix in my notebook; discussed it with the team |
| 4 | A teammate asked to move the code into notebooks and unify the results files | `baseline.ipynb` and `save_results()` in `ranking_metrics.py` (shared JSON format with a `description` field) | Replaced `src/` with the notebook, re-ran it, confirmed that the numbers match the earlier runs |
| 5 | Asked to compare other models, use skill aliases, and do fine-tuning | `experiments/01_models_and_aliases.ipynb` (MiniLM, MPNet, JobBERT-v2; aliases first/mean/max); | Ran both on my Mac, reported the tables and the output of the `SMOKE` mode |
| 6 | Asked for error analysis and interpretation of the output | Interpretation of per-job metrics, core-vs-contextual AUC, examples of errors | Ran the notebook, saved the outputs; |
| 7 | Asked about limitations (dev vs official test; the model's training data) | Explained how the candidate pool was built and the label mismatch after reading the overview paper | |


## Errors of the LLM that were detected

- An error in the first version of `evaluate_ranking.py` (text encoding was called inside
  a list, i.e. thousands of times) was found by the LLM itself while reviewing its own code;
  I ran the script and fixed it.
- A statistic about the mismatch between the training labels (essential/optional) and the evaluation
  grades (core/contextual) was first attributed to the organizers; after reading the
  overview paper it turned out to be a measurement reported by one of the participating teams
  (bipboopbipboop). The attribution was corrected.
- The LLM's recommendation of a literature source for JobBERT-v2 was uncertain;
  the sources disagree on which paper describes it.
- The LLM could not run the fine-tuning code; it
  was verified only with stubs. I ran and verified the real training myself.

## What the LLM did not do

- It had no access to my repository or computer; it saw only what I pasted into the chat.
- The experimental results in the report come from my own runs
  (`results/*.json` are reproducible with the notebooks), not from the LLM.


  # Use of LLMs and AI Agents — [Name ]


  # Use of LLMs and AI Agents — [Name ]



  # Use of LLMs and AI Agents — [Name ]



