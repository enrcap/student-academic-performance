import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold


def _slice_rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices].copy()
    return X[indices]


def _slice_target(y, indices):
    if hasattr(y, "iloc"):
        return y.iloc[indices].copy()
    return y[indices]


def cross_validation_supervised(
    model_factory,
    X,
    y,
    preprocessor,
    n_folds: int = 5,
    task: str = "classification",
    scorers: dict | None = None,
    fit_params: dict | None = None,
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Validacion cruzada general para modelos supervisados del proyecto.

    Adaptada de la idea de la p2-6, pero permitiendo:
    - clasificación estratificada
    - regresion con KFold
    - preprocesado ajustado dentro de cada fold
    - multiples metricas personalizadas
    """
    if scorers is None or len(scorers) == 0:
        raise ValueError("Debes proporcionar al menos una metrica en `scorers`.")

    if task not in {"classification", "regression"}:
        raise ValueError("task debe ser 'classification' o 'regression'.")

    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state,
        )
        split_iterator = splitter.split(X, y)
    else:
        splitter = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state,
        )
        split_iterator = splitter.split(X)

    fit_params = {} if fit_params is None else fit_params.copy()
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
        X_train = _slice_rows(X, train_idx)
        X_valid = _slice_rows(X, valid_idx)
        y_train = _slice_target(y, train_idx)
        y_valid = _slice_target(y, valid_idx)

        fold_preprocessor = clone(preprocessor)
        X_train_prepared = fold_preprocessor.fit_transform(X_train)
        X_valid_prepared = fold_preprocessor.transform(X_valid)

        model = model_factory()
        y_train_values = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        y_valid_values = y_valid.to_numpy() if hasattr(y_valid, "to_numpy") else np.asarray(y_valid)

        model.fit(X_train_prepared, y_train_values, **fit_params)
        y_pred = model.predict(X_valid_prepared)

        row = {"fold": fold}
        for metric_name, scorer in scorers.items():
            row[metric_name] = float(scorer(y_valid_values, y_pred))
        rows.append(row)

    fold_scores = pd.DataFrame(rows)
    summary = (
        fold_scores.drop(columns="fold")
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    return fold_scores, summary


def cross_validation_clustering(
    X,
    preprocessor,
    n_clusters: int,
    n_folds: int = 5,
    pca_variance_to_keep: float = 0.85,
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Validacion cruzada para clustering mediante silhouette en folds de validacion.

    Flujo por fold:
    1. Ajustar preprocesado con train.
    2. Ajustar PCA con train.
    3. Ajustar KMeans con train.
    4. Predecir clusters en validacion.
    5. Calcular silhouette en validacion.
    """
    splitter = KFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )

    rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train = _slice_rows(X, train_idx)
        X_valid = _slice_rows(X, valid_idx)

        fold_preprocessor = clone(preprocessor)
        X_train_prepared = fold_preprocessor.fit_transform(X_train)
        X_valid_prepared = fold_preprocessor.transform(X_valid)

        pca = PCA(
            n_components=pca_variance_to_keep,
            svd_solver="full",
            random_state=random_state,
        )
        X_train_pca = pca.fit_transform(X_train_prepared)
        X_valid_pca = pca.transform(X_valid_prepared)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        )
        kmeans.fit(X_train_pca)
        valid_labels = kmeans.predict(X_valid_pca)

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        if len(unique_labels) < 2 or np.min(counts) < 2:
            silhouette = np.nan
        else:
            silhouette = float(silhouette_score(X_valid_pca, valid_labels))

        rows.append(
            {
                "fold": fold,
                "silhouette": silhouette,
                "n_components": int(pca.n_components_),
            }
        )

    fold_scores = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": "silhouette",
                "mean": float(fold_scores["silhouette"].mean(skipna=True)),
                "std": float(fold_scores["silhouette"].std(skipna=True)),
                "valid_folds": int(fold_scores["silhouette"].notna().sum()),
            }
        ]
    )
    return fold_scores, summary
