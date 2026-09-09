from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


PROJECT_PALETTE = {
    "primary": "#2B6CB0",
    "secondary": "#2F855A",
    "accent": "#D97706",
    "danger": "#C53030",
    "muted": "#4A5568",
}


def ensure_dir(path: str | Path) -> Path:
    """Crea una carpeta si no existe y devuelve la ruta."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_project_style() -> None:
    """Aplica un estilo comun para las figuras del proyecto."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def adaptar_dataframe_a_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte pd.NA y dtypes anulables a un formato robusto para sklearn.

    - Las columnas numericas pasan a float con np.nan como faltante.
    - Las columnas no numericas se mantienen como object con np.nan.
    """
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        else:
            out[col] = out[col].astype(object)
            out.loc[pd.isna(out[col]), col] = np.nan

    return out


def plot_confusion_heatmap(
    y_true,
    y_pred,
    labels,
    title: str,
    normalize: bool = True,
    cmap: str = "Blues",
    ax=None,
):
    """Dibuja una matriz de confusion como mapa de calor."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    return ax


def annotate_bars(ax, decimals: int = 3) -> None:
    """Anade etiquetas numericas encima de un grafico de barras."""
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height) or patch.get_width() == 0:
            continue
        ax.annotate(
            f"{height:.{decimals}f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )
