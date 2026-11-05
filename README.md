# JOptim: Unpacking Interpretability

Analysis and experiment code accompanying the article *Unpacking Interpretability: Human-Centered Criteria for Optimal Combinatorial Solutions* (CHBR-D-26-00486R1).
All data inputs are available at <https://doi.org/10.5281/zenodo.23964950>; the deposit is laid out like the code's data root (see *Data* below).

## Setup

```
git clone --recurse-submodules https://github.com/univiemops/joptim-analysis.git
cd joptim-analysis
conda env create -n jo -f environment.yml && conda activate jo
pip install -e .
pip install -e ./pub-utils        # pinned via the submodule
```

Create a `.env` in the repository root (the file is ignored by git; `env_dev` is a template):

```
JOPTIM_API_KEY=your-secret-download-uri
JOPTIM_SERVER_URL=http://localhost
```

The pipelines locate the data root through the file `~/code/data_export_dir`, whose first line is the path to the parent directory of a `joptim-analysis/` data folder (containing `data/`, `pub/`, `logs/`; see `src/joptim/utils/export_path.py`).

`JOPTIM_API_KEY` is the access token of the experiment server's download API; it is only needed by the data-download notebooks. The data used in the article are available in the Zenodo deposit.

## Structure

- `notebooks/experiment/` — generation and filtering of problem instances and solutions, trial generation and sanity checks
- `notebooks/data_processing/` — downloading experiment data (server API and Prolific), preprocessing, eye-tracking data handling
- `notebooks/parameter_fitting/` — fitting the parameters of the composition-complexity model
- `notebooks/analysis/` — the analyses of the article: demographics, multilevel choice/RT/gaze models, coherence checks, descriptives, forest plots; every notebook has a jupytext-paired mirror under `py/` that is the canonical source
- `src/joptim/` — package code (data wrangling, export paths, metrics)
- `tools/build_deposit_zips.py` — builds the upload-ready Zenodo deposit
- `pub-utils/` — submodule with publication figure/table helpers

## Workflow (step by step)

### A. Exploratory phase (`STUDY="exploratory"`)

1. **Create problems and initial solution metrics** — `notebooks/experiment/generate_problems_solutions.ipynb` (output: `problems.pkl`), then `notebooks/experiment/extend_filter_solutions.ipynb` (computes initial composition-complexity values using parameters found via PCA fitting; these parameters are set in the notebook, not loaded from a pickle; output: `problems_filtered_incl_solution_metrics.pkl`).
2. **Generate trials** — `python notebooks/experiment/generate_trials.py --exploratory` (consumes the filtered problems and solutions, creates trial pairings; output: `trials_exploratory.pkl`), then `notebooks/experiment/trials_sanity_checks.ipynb` to validate integrity before deploying the experiment.
3. **Run the experiment and collect data.**
4. **Preprocess and fit parameters** — `notebooks/data_processing/preprocess_experiment_data.ipynb` (output: preprocessed results CSVs), then `notebooks/parameter_fitting/mixture_model_fit_pilot.ipynb` (finds the best parameters for composition complexity v2; stores them in a pickle, e.g. `optimization_results_composition_to_pilot_cc_v2.pkl`).

### B. Confirmatory phase (`STUDY="confirmatory"`)

5. **Update problems with confirmatory composition complexity** — `notebooks/experiment/extend_solutions_for_confirmatory_analysis.ipynb` (loads the best-fit parameters from the stored pickles and recomputes composition complexity for all solutions with a `_confirmatory` suffix; **must run before trial generation**).
6. **Generate confirmatory trials** — `python notebooks/experiment/generate_trials.py --confirmatory`, then the sanity checks.
7. **Run the confirmatory experiment and analyze** — repeat the preprocessing and the analyses (choice, reaction time, gaze) with the confirmatory results.

### Notes

- The PCA-derived parameters for the initial exploratory solution extension are set manually; do not load them from a pickle.
- The confirmatory composition complexity parameters are loaded from the stored pilot fit pickle.
- The metric value used in trial generation (`diff_cc`, etc.) is always sourced from the last metric computed in the problems pickle.

## Publication assets

The analysis notebooks write their JSON metrics, figures, and tables to the data root's `pub/` tree.
`tools/build_deposit_zips.py` packages `data/`, `stimuli/`, and `stimuli_pairs/` into the Zenodo deposit zips.

## Data layout

The data root contains:

- `data/` — raw session tables (`results*.pkl`), problem pools (`problems*.pkl`), preprocessed CSVs, per-study trial definitions, Prolific demographic tables, free-text strategy responses, and descriptives/model JSON outputs
- `stimuli/`, `stimuli_pairs/` — stimulus material
- `experiment_interface/`, `logs/`, `plots/`, `OSF/` — auxiliary outputs

Note: the Zenodo deposit is de-identified (participant-identifier columns are removed), so the participant-level pipeline steps expect the original working data.

## Contact

Maintainer: [dominikpegler](https://github.com/dominikpegler)
