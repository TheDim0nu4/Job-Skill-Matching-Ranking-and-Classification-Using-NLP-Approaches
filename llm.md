# Use of LLMs and AI Agents — Anatolii Vasylov, embeddings part

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


# Use of LLMs and AI Agents — Valeriia Nikolaieva, methodology and documentation part

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


# Use of LLMs and AI Agents — Danil Kozhan, final experiment pipeline

This section documents use of an AI assistant while implementing, running,
debugging, validating, and documenting the final experiment pipeline in
Notebooks 04--07.

## Tool

- **Assistant:** AI assistant using Copilot SDK in VS Code
- **Mode of use:** interactive coding, analysis, debugging, result
  interpretation
- **Verification:** source files, notebooks, generated result files, local
  execution output, and the official Task B description were used as sources
  of truth

The assistant did not independently produce or invent the reported experiment
metrics. I ran the notebooks locally, inspected their outputs, and made the
final decisions about the implementation and interpretation.

## General role of the LLM

I used the assistant to help turn the exploratory work into a controlled final
pipeline. It helped review the repository and task requirements, implement and
refine Notebooks 04--07, diagnose execution errors, check metric and dataset
consistency.

The assistant's suggestions were checked against the implementation and actual
outputs. The repository result files remained authoritative for numerical
claims.

## Log by stage

| # | What I asked (summary) | What the LLM produced | What I did / verified |
|---|---|---|---|
| 1 | Implemented the five-model and three-alias comparison in Notebook 04 | Code for evaluating MiniLM, MPNet, JobBERT, BGE, and E5 with first, mean, and max alias representations; result aggregation and alias-strategy summaries | Ran all 15 configurations and verified the generated result files and summary metrics |
| 2 | Implemented the locked comparison and model-selection stage in Notebook 05 | Selection logic using mean graded nDCG across models, with MAP as the tie-breaker; locked comparison and fine-tuning candidate export | Verified that mean aliases were selected and that JobBERT, E5, and MPNet were selected for fine-tuning |
| 3 | Implemented fine-tuning of the selected models in Notebook 06 | A contrastive fine-tuning workflow using `MultipleNegativesRankingLoss`, positive pairs, fixed evaluation data, and reproducible configuration metadata | Ran fine-tuning locally for JobBERT, E5, and MPNet and verified the saved checkpoints and result files |
| 4 | Implemented the final comparison and error analysis in Notebook 07 | Code combining five zero-shot and three fine-tuned systems, calculating aggregate and per-job metrics, AUC values, and hardest-job exports | Ran the notebook and verified 2,432 per-job rows, eight system summaries, and the generated error-analysis files |


## Errors and verification

- The first JobBERT fine-tuning route configuration used route names that
  were not available in the model. The error was investigated and the working
  `anchor`/`positive` mapping was verified by a successful local run.
- Dataset counts and processed-row totals were checked against the current
  repository files rather than copied from earlier documentation.


## What the LLM did not do

- It did not generate the measured retrieval metrics or replace the local
  notebook runs.
- It did not independently validate results without access to the repository
  files and execution outputs.
- The final numerical values were taken from the generated result files and
  checked against the implementation.



# Use of LLMs and AI Agents — Dmytro Skrypchenko, data preparation

This section documents how I used an LLM while preparing the Task B data and exploring negative sampling.

## Tool

- **Assistant:** ChatGPT (OpenAI), via the ChatGPT interface
- **Mode of use:** interactive coding assistance, explanations, and code review
- **Verification:** the project's data files and notebook outputs were used to guide data-preparation decisions.

The assistant provided suggestions and explanations in response to code and questions I supplied. I remained responsible for deciding which suggestions to use and for running and checking the code locally.

## General role of the LLM

I used the LLM as a programming and methodology assistant. It helped me structure the training and validation dataframes, add synthetic negative job–skill pairs, understand the assumptions behind negative sampling, and prepare the data for subsequent experiments.

## Log by stage

| # | What I asked (summary) | What the LLM produced | What I did / verified |
|---|---|---|---|
| 1 | Asked for a simple way to generate 50,000 random negative job–skill pairs for the training data | A Python approach using `random`, sets of known positive pairs, and batched sampling | Reviewed the code and adapted it to the notebook's existing dataframe and column names |
| 2 | Asked how to add generated negative pairs to the training dataframe | Code to convert the pairs into a dataframe, join job and skill names, assign the synthetic `irrelevant` label, and concatenate them with the existing data | Integrated the suggested dataframe operations into the data-preparation workflow |
| 3 | Asked how to prepare a corresponding dataset from the `validation` files | Guidance for loading the validation queries, candidate skills, and qrels, then joining IDs to their text representations | Used the actual validation file structure and adjusted the parsing and column names to match the files |
| 4 | Asked about an error while parsing `skill_aliases` with `json.loads()` | An explanation that the values were not necessarily valid JSON and a suggestion to use `ast.literal_eval()` for Python-literal list strings | Applied the alternative parsing method in the notebook |

**Note:** The generated negative pairs are synthetic examples. An unlabeled job–skill pair is not necessarily truly irrelevant, so this assumption should be considered when interpreting experimental results.
