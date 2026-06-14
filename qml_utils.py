"""qml_utils.py

Shared helper functions for the Quantum Machine Learning Classifier project:
"Encoding Strategies for Quantum Machine Learning Classification".

These helpers keep the seven phase notebooks consistent so that every
encoding strategy is compared on exactly the same data, the same train/test
split, and the same plotting style. Keeping this code in one place lets the
notebooks focus on the quantum ideas instead of repeating boilerplate.

Sections
--------
1. LaTeX rendering helper for NumPy arrays in Jupyter notebooks
2. Reproducibility
3. Dataset loading and preprocessing
4. Amplitude-encoding data preparation
5. Plotting helpers (dataset and decision boundaries)
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Math
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Section 1: LaTeX rendering

# (value, LaTeX string) pairs for recognizing common exact fractions.
# Comparisons are made against abs(np.round(val, 5)).
_SPECIAL_VALUES: list[tuple[float, str]] = [
    (0.25000, r"\frac{1}{4}"),
    (0.50000, r"\frac{1}{2}"),
    (0.57735, r"\frac{1}{\sqrt{3}}"),
    (0.70711, r"\frac{1}{\sqrt{2}}"),
    (0.81650, r"\sqrt{\frac{2}{3}}"),
    (0.86603, r"\frac{\sqrt{3}}{2}"),
]


def _format_real(val: float, places: int = 5) -> str:
    """Return a LaTeX string for a real scalar, recognizing common fractions"""
    rounded = abs(np.round(val, 5))
    for threshold, latex in _SPECIAL_VALUES:
        if rounded == threshold:
            return ("-" if val < 0 else "") + latex

    fmt = f"{{v:.{places}f}}"
    result = fmt.format(v=val).rstrip("0").rstrip(".")
    return "0" if float(result) == 0 else result


def as_latex(
    a: np.ndarray,
    places: int = 5,
    column: bool = False,
    prefix: str = "",
) -> Math:
    """Render a NumPy array as a LaTeX bmatrix for display in a Jupyter notebook

    Args:
        a:       1-D or 2-D array of real or complex values
        places:  Decimal places used when formatting non-special values
        column:  If True, treat a 1-D array as a column vector
        prefix:  Optional LaTeX string prepended before the bmatrix

    Returns:
        An IPython ``Math`` object ready for ``display()``
    """
    matrix = np.copy(a)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
        if column:
            matrix = matrix.T

    precision = 1 / 10**places
    rows: list[str] = []

    for row in range(matrix.shape[0]):
        cells: list[str] = []
        for col in range(matrix.shape[1]):
            cell = matrix[row, col]
            real_part = float(np.real(cell))
            imag_part = float(np.imag(cell))

            is_imag_neg = imag_part < 0
            is_real_zero = np.isclose(real_part, 0, atol=precision)
            is_imag_zero = np.isclose(imag_part, 0, atol=precision)
            is_imag_one = np.isclose(abs(imag_part), 1, atol=precision)

            cell_parts: list[str] = []
            if is_real_zero and is_imag_zero:
                cell_parts.append("0")
            elif not is_real_zero:
                cell_parts.append(_format_real(real_part, places))

            if not is_imag_zero:
                if is_imag_one:
                    cell_parts.append(
                        "-i" if is_imag_neg else ("+" if not is_real_zero else "") + "i"
                    )
                else:
                    if not is_real_zero and not is_imag_neg:
                        cell_parts.append(" + ")
                    cell_parts.append(_format_real(imag_part, places) + "i")

            cells.append("".join(cell_parts))

        rows.append(" & ".join(cells))

    latex_body = r" \\[1em] ".join(rows)
    return Math(prefix + r"\begin{bmatrix}" + latex_body + r"\\" + r"\end{bmatrix}")


# Section 2: Reproducibility

# A single fixed seed is used everywhere so results are repeatable. The same
# value is also assigned to Qiskit's algorithm_globals.random_seed inside the
# notebooks, which controls the random starting parameters of the ansatz.
RANDOM_SEED: int = 42


# Section 3: Dataset loading and preprocessing


def load_moons(
    n_samples: int = 200,
    noise: float = 0.20,
    test_size: float = 0.30,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the two-moons dataset and split it into train and test sets.

    The two-moons dataset is a classic non-linearly-separable benchmark: two
    interleaving half circles. It is ideal here because it cannot be split by
    a straight line, so the choice of feature encoding genuinely matters, and
    because it is two-dimensional, which lets us draw decision boundaries.

    Parameters
    ----------
    n_samples : int
        Total number of points to generate.
    noise : float
        Standard deviation of Gaussian noise added to the data.
    test_size : float
        Fraction of the data held out for testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x_train, x_test, y_train, y_test (features are unscaled here).
    """
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    return x_train, x_test, y_train, y_test


def scale_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_range: tuple[float, float] = (0.0, np.pi),
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale features into a fixed range using statistics from the training set.

    Angle and ZZ feature maps interpret each feature value as a rotation angle.
    Scaling into [0, pi] keeps those angles in a well-behaved range and makes
    sure the test set is transformed using only training-set statistics (no
    data leakage).

    Parameters
    ----------
    x_train : np.ndarray
        Training features (fit the scaler on these).
    x_test : np.ndarray
        Test features (transform only).
    feature_range : tuple[float, float]
        Target range for the scaled features.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, MinMaxScaler]
        Scaled x_train, scaled x_test, and the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


# Section 4: Amplitude-encoding data preparation


def pad_and_normalize(x: np.ndarray, num_amplitudes: int = 4) -> np.ndarray:
    """Prepare classical data for amplitude encoding.

    Amplitude encoding stores a feature vector directly in the amplitudes of a
    quantum state. Two requirements follow from that:

    1. The number of amplitudes must be a power of two (here 4 -> 2 qubits),
       so we zero-pad each 2-feature sample up to length 4.
    2. A quantum state must have unit length, so each padded vector is
       L2-normalized.

    Normalization means amplitude encoding keeps only the *direction* of the
    feature vector, not its overall magnitude. That is an honest limitation of
    the method and is discussed in the notebooks.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_features).
    num_amplitudes : int
        Length to pad each sample to (must be a power of two).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, num_amplitudes) with unit-norm rows.
    """
    n_samples, n_features = x.shape
    padded = np.zeros((n_samples, num_amplitudes))
    padded[:, :n_features] = x
    norms = np.linalg.norm(padded, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid dividing by zero for an all-zero row
    return padded / norms


# Section 5: Plotting helpers


def plot_dataset(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter-plot a two-dimensional, two-class dataset."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


def plot_decision_boundary(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Decision boundary",
    ax: plt.Axes | None = None,
    grid_steps: int = 50,
    margin: float = 0.1,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    cmap: str = "coolwarm",
) -> plt.Axes:
    """Draw a model's decision boundary over a 2-D feature space.

    A regular grid of points is built across the plotting region, each grid
    point is classified, and the predicted classes are shown as a colored
    background. The training/test points are drawn on top.

    Parameters
    ----------
    predict : Callable[[np.ndarray], np.ndarray]
        A function mapping an (n, 2) array to a length-n array of class labels
        (for example svc.predict or vqc.predict).
    x : np.ndarray
        The 2-D points to scatter on top (in the same coordinates as the grid).
    y : np.ndarray
        Class labels for the scattered points.
    title : str
        Plot title.
    ax : plt.Axes | None
        Existing axis to draw on, or None to create one.
    grid_steps : int
        Number of grid points per axis. Larger is smoother but slower, which
        matters for quantum models that run a circuit per prediction.
    margin : float
        Extra space added around the data range.
    transform : Callable[[np.ndarray], np.ndarray] | None
        Optional transform applied to the grid points before prediction. Used
        for amplitude encoding, where 2-D grid points must be padded and
        normalized before the classifier can accept them.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    plt.Axes
        The axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    x_min, x_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    y_min, y_max = x[:, 1].min() - margin, x[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_for_model = transform(grid) if transform is not None else grid
    z = np.asarray(predict(grid_for_model)).reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.3, cmap=cmap)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, edgecolors="k", s=25)
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax
