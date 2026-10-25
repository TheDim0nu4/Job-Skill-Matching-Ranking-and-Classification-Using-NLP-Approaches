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


# Use of LLMs and AI Agents — [Valeriia Nikolaieva], methodology and documentation part

This appendix documents how I used an LLM while analysing the TalentCLEF
Task B project, defining the experimental methodology, interpreting the
existing results, and preparing the project documentation.

## Tool

- **Assistant:** AI assistant using Copilot SDK in VS Code
- **Mode of use:** interactive analysis and writing assistance; no autonomous
  agent independently ran experiments or committed changes on my behalf
- **Verification:** repository files, notebooks, saved result tables, and
  official task documentation were used as the sources of truth

## General role of the LLM

I used the LLM as a research and documentation assistant. It helped me read
the official TalentCLEF Task B description, analyse the existing repository,
interpret the baseline and fine-tuning experiments, compare metrics, identify
methodological limitations, and structure the final report.

The LLM did not generate the experimental results. The reported numbers were
read from the repository's notebooks and result files, especially
`results/final_system_comparison.csv`,
`results/error_analysis_summary.csv`, and
`results/alias_strategy_summary.csv`. I made the final decisions about the
methodology, checked the suggested explanations against the implementation,
and requested corrections whenever the documentation did not match the
current code or data.

## Log by stage

| # | What I asked (summary) | What the LLM produced | What I did / verified |
|---|---|---|---|
| 1 | Sent the official TalentCLEF Task B description and the course requirements for the first submission | A structured interpretation of the task: retrieve ESCO skills for a job title, rank them, and distinguish core (`2`) from contextual (`1`) relevance through graded judgments | Compared the interpretation with the official Task B data description and used it to define the project's problem, goals, and evaluation requirements |
| 2 | Asked what should be done next after the initial experiments | A proposal to formalise the final baseline, compare models under one protocol, add a lexical reference baseline, analyse core/contextual behaviour, and document false-negative risks | Used the proposal to organise the later methodology around the final notebooks 04--07, while keeping proposed extensions separate from implemented features |
| 3 | Reviewed all repository changes and the final experiment pipeline | A summary of the transition from the early experiments to the final five-model comparison, locked alias selection, selected-model fine-tuning, final comparison, and per-job error analysis | Verified the commit history, notebook contents, result CSV files, and the current README/results documentation; treated notebooks 04--07 as the final methodology rather than notebooks 01--03 |
| 4 | Asked for a complete Slovak academic report | A first structured draft of the Slovak LaTeX report covering the problem, related work, data, preprocessing, final methodology, metrics, results, AI usage, reproducibility, and references | Manually checked the report against the current repository, corrected JSON and qrels counts, corrected the JobBERT routing description, aligned the final table with `final_system_comparison.csv`, and removed claims not supported by the implementation |
| 5 | Asked for a final review of the report and the generated PDF/source | A checklist-based review of numerical consistency, citation keys, qrels semantics, retrieval-versus-classification wording, reproducibility claims, AUC interpretation, and typography | Personally reviewed and corrected `docs.tex` and `refs.bib`; verified all cited BibTeX keys, all final metrics, the execution order, the absence of an explicit classifier, and the removal of the dedicated limitations section |

## Literature and task analysis

The LLM helped me analyse the official Task B description and the papers
provided in the repository's `refs/` directory. This analysis was used to
explain the task as an information-retrieval problem with graded relevance,
rather than as a conventional three-class classification problem. It also
helped compare the project's bi-encoder/embedding approach with the hybrid
ESCO-aware and ensemble bi-encoder approaches described by previous
TalentCLEF participants.

The resulting report cites the sources through `refs.bib`. I checked that
each in-text citation has a matching BibTeX entry and that the final report
does not present broader terms such as complementary or transversal as
implemented labels. The implemented development labels are explicitly
described as core (`2`) and contextual (`1`), while candidates absent from
qrels receive grade `0` during evaluation.

## Documentation contribution

I prepared and revised the project documentation with LLM assistance, using
it to help connect the data preparation, model comparison, alias selection,
fine-tuning, final evaluation, error analysis, and reproducibility procedure.
I manually checked the generated text, corrected mistakes, and made the final
decisions about what was included.
The documentation was checked against:

- `README.md`;
- `report.md`;
- `results.md`;
- `ranking_metrics.py`;
- `config.py`;
- `requirements.txt`;
- the final experiment notebooks;
- the authoritative result CSV files.

The LaTeX report `docs.tex` was written in Slovak and includes citations from
`refs.bib`, exact current data counts, the final metrics table, the execution
order of the notebooks, and the distinction between implemented behaviour and
future work. The LLM helped with language, structure, and consistency checks,
but the repository files remained the authority for all technical claims and
numerical values.

## Verification and boundaries

- I did not treat LLM-generated explanations as experimental evidence.
- I checked numerical claims against the current CSV files instead of
  copying earlier values from conversation history.
- I corrected an unresolved BibTeX key and added missing publication years to
  the participant-paper entries.
- I explicitly documented that fine-tuning treats labels `1` and `2` as
  positive pairs and does not implement a separate core/contextual
  classifier.
- I qualified claims about AUC because candidates absent from qrels are
  assigned grade `0` during evaluation.
- I did not claim statistical significance because no significance tests were
  performed.
