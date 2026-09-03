# QNDM-VQC ECG Classification

Publication-scale experiment code for evaluating QNDM-style gradient estimation in variational quantum classifiers on MIT-BIH ECG heartbeat classification.

The project compares analytic gradients, finite-shot parameter-shift gradients, and finite-shot QNDM detector-gradient estimates for an AAMI N-vs-V heartbeat classification task. It includes data preprocessing, patient-independent splitting, PCA feature extraction, variational quantum circuit training, gradient validation, shot/lambda sweeps, classical baselines, resource accounting, plotting, and report generation.

## Research Question

The main question is whether QNDM gradient estimation provides a practical training or measurement-resource advantage over direct parameter-shift gradients in a small variational quantum ECG classifier.

This repository is designed to make that question testable and reproducible. It does not silently replace failed QNDM estimates with parameter-shift gradients, and it preserves negative or inconclusive results.

## Repository Contents

```text
configs/                  Experiment configurations: quick, standard, full
src/data/                 MIT-BIH loading, beat extraction, preprocessing, PCA features
src/quantum/              VQC ansatz, observables, gradients, parameter shift, QNDM detector logic
src/training/             Training loop, optimizer, evaluation, shot allocation interface
src/experiments/          Gradient validation, lambda sweep, shot sweep, training experiments
src/analysis/             Statistics, plots, tables, summary report helpers
tests/                    Unit and integration tests
scripts/                  Utility scripts, including LaTeX package generation
run_experiments.py        Main experiment runner
run_all.sh                One-command setup, test, and run wrapper
```

Generated data, plots, model parameters, raw MIT-BIH files, virtual environments, caches, and LaTeX result packages are intentionally not tracked by Git.

## Dataset

This code expects the MIT-BIH Arrhythmia Database to be available locally. By default, place the database at:

```text
MIT-BIH Arrhythmia/
```

The directory should contain the WFDB record files such as `.dat`, `.hea`, and `.atr`.

The dataset is not redistributed in this repository. You can either place the files manually or run the experiment runner with `--download-data` if `wfdb` is installed and network access is available.

## Task Definition

The classification task is AAMI N-type versus V-type heartbeat classification:

| Class | MIT-BIH Beat Labels |
|---|---|
| N | `N`, `L`, `R`, `e`, `j` |
| V | `V`, `E` |

The default split is patient independent:

- DS1 records are used for training and validation.
- DS2 records are held out for final testing.
- Paced records `102`, `104`, `107`, and `217` are excluded.
- Scaling and PCA are fit only on the training portion to avoid leakage.

## Methods

The variational quantum classifier uses angle-encoded ECG features, trainable `RY` rotations, and nearest-neighbor CNOT entangling layers.

Implemented gradient/evaluation modes:

- exact analytic gradients from the statevector simulator;
- parameter-shift gradient validation;
- finite-shot parameter-shift training;
- finite-lambda QNDM detector-gradient validation;
- finite-shot QNDM-gradient training;
- matched-precision shot/resource analysis;
- classical baselines on the same PCA features.

## Installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Running Experiments

Quick smoke test:

```bash
python -u run_experiments.py --config configs/quick.yaml
```

Standard run:

```bash
bash run_all.sh standard
```

Full publication-scale run:

```bash
bash run_all.sh full
```

To force a fresh full run:

```bash
python -u run_experiments.py --config configs/full.yaml --force
```

The runner prints progress messages and writes all artifacts to `output/`.

## Configurations

| Config | Purpose |
|---|---|
| `configs/quick.yaml` | Minimal smoke test for pipeline correctness |
| `configs/standard.yaml` | Moderate run for development and debugging |
| `configs/full.yaml` | Full 4/6/8-qubit, multi-seed publication-scale run |

The full configuration evaluates:

- qubits: 4, 6, 8;
- primary layers: 2;
- methods: analytic, parameter shift, QNDM;
- seeds: 42, 123, 456, 789, 2026;
- lambda sweep: `0.0001` to `0.1`;
- shot sweep: 100 to 5000 shots;
- classical reference baselines.

## Outputs

Experiment outputs are generated under:

```text
output/
```

Typical outputs include:

- dataset summaries and split checks;
- PCA feature summaries;
- gradient correctness tests;
- lambda and shot sensitivity sweeps;
- final classification metrics;
- confusion matrices;
- training and validation curves;
- resource-comparison tables;
- classical baseline metrics;
- validation and research summary reports.

These files are ignored by Git because they can be large and are reproducible from the code and configuration.

## LaTeX Report Package

After running experiments, a compact LaTeX package can be generated locally:

```bash
python scripts/build_latex_package.py
```

This creates:

```text
output/qndm_ecg_latex_package.zip
output/latex_package/main.tex
output/latex_package/figures/
```

The LaTeX package is not committed to GitHub.

## Tests

Run the test suite with:

```bash
pytest -q
```

The tests cover:

- dataset detection and label mapping;
- patient-independent split integrity;
- train-only PCA fitting;
- parameter-shift identity;
- QNDM detector execution;
- QNDM small-lambda agreement;
- resource-count equations;
- pipeline output creation and reproducibility.

## Notes On Interpretation

The code is intended for controlled research experiments, not clinical deployment. Any empirical claim should be based on regenerated outputs, validation checks, and statistical analysis from the exact configuration used.

In the completed local full run used during development, analytic and finite-shot parameter-shift training behaved consistently, while the tested finite-shot QNDM training configuration was unstable. That result should be treated as a negative or diagnostic finding unless the finite-shot QNDM protocol is redesigned and independently validated.

## License

No license has been specified yet. Add a license before public reuse or redistribution.
