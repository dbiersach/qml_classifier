# Encoding Strategies for Quantum Machine Learning Classification

A standalone workspace for a comparative study of three quantum feature-encoding
strategies (angle encoding, amplitude encoding, and ZZ feature map encoding)
inside variational quantum classifiers (VQCs), measured against a classical
RBF-kernel support vector machine (SVM) baseline.

Everything runs locally on Qiskit's `AerSimulator`. No IBM Quantum account or
real quantum hardware is required.

## Hypothesis

> Encoding strategies that incorporate quantum entanglement (the ZZ feature
> map) will produce measurably different decision boundaries than a
> non-entangling encoding (angle encoding), but no quantum encoding is
> expected to surpass a classical RBF-kernel SVM on small benchmark datasets.

A null result against the classical baseline is a valid scientific finding.

## Setup

This workspace uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment.

```bash
# From the workspace folder, create the virtual environment from uv.lock
uv sync
```

Then open the folder (or `qml_classifier.code-workspace`) in VS Code and select
the `.venv` interpreter / the project Jupyter kernel. Run the notebooks in
order with all cells top to bottom.

## The phases

| Notebook | Phase | What you do |
| --- | --- | --- |
| `01_setup_and_research_question.ipynb` | Setup | Confirm the software stack, state the research question and roadmap |
| `02_datasets_and_classical_baseline.ipynb` | Baseline | Load the two-moons data, train the classical RBF SVM, record the benchmark |
| `03_quantum_encodings_explained.ipynb` | Encodings | Visualize all three encodings as circuits and quantum states (no training) |
| `04_vqc_angle_encoding.ipynb` | VQC #1 | Train a VQC using angle encoding (no entanglement) |
| `05_vqc_amplitude_encoding.ipynb` | VQC #2 | Train a VQC using amplitude encoding |
| `06_vqc_zz_feature_map.ipynb` | VQC #3 | Train a VQC using the ZZ feature map (entangling) |
| `07_comparative_analysis.ipynb` | Analysis | Retrain all four models together and evaluate the hypothesis |

`future_quantum_advantage.md` is a companion discussion note: when and why the
quantum approach could become attractive for larger or higher-dimensional
problems.

## Shared code

`qml_utils.py` holds the dataset loading, preprocessing, plotting, and LaTeX
rendering helpers so every notebook compares the encodings on identical data,
the same train/test split, and the same plotting style. It sits beside the
notebooks and is imported directly (`import qml_utils as hu`), so no package
install is needed.

## Design choices that keep the comparison fair

- **One dataset:** the two-moons benchmark (200 samples), which is
  non-linearly separable and two-dimensional so decision boundaries can be
  drawn.
- **One split and seed:** a fixed random seed (42) and a single stratified
  70/30 train/test split, applied identically across notebooks.
- **One ansatz and optimizer:** every VQC uses the `real_amplitudes` ansatz
  and the gradient-free `COBYLA` optimizer, so the **encoding is the only
  variable** that changes between Phases 4, 5, and 6.
- **Honest amplitude encoding:** 2 features are zero-padded to 4 amplitudes
  and L2-normalized (2 qubits). Normalization discards magnitude, which is a
  real and discussed limitation of the method.

## Software

Qiskit 2.x, `qiskit-aer`, `qiskit-machine-learning`, scikit-learn, NumPy,
Matplotlib, and Pandas, pinned in `uv.lock`. The notebooks follow the modern
functional Qiskit API (`z_feature_map`, `zz_feature_map`, `real_amplitudes`,
`raw_feature_vector`), since the older class-based forms are deprecated in
Qiskit 2.x.

## Coding style

See `AGENTS.md` (and the identical `.github/copilot-instructions.md`) for the
notebook and Python style guide this project follows.
