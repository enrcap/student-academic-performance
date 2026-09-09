import pandas as pd


TARGET_COL = "nota_media_2sem"

SEM1_COLS = [
    "asignaturas_1sem_convalidadas",
    "asignaturas_1sem_matriculadas",
    "asignaturas_1sem_evaluadas",
    "asignaturas_1sem_aprobadas",
    "nota_media_1sem",
    "asignaturas_1sem_sin_evaluacion",
]

SEM2_COLS = [
    "asignaturas_2sem_convalidadas",
    "asignaturas_2sem_matriculadas",
    "asignaturas_2sem_evaluadas",
    "asignaturas_2sem_aprobadas",
    "nota_media_2sem",
    "asignaturas_2sem_sin_evaluacion",
]


def get_regression_dataset(
    df: pd.DataFrame,
    stage: str = "sem1",
    drop_missing_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y para regresion evitando fuga de informacion del 2o semestre.

    stage='sem1' mantiene variables seguras del primer semestre.
    stage='early' elimina tambien las variables del primer semestre para una
    prediccion mas temprana.
    """
    if stage not in {"sem1", "early"}:
        raise ValueError("stage debe ser 'sem1' o 'early'.")

    drop_cols = ["objetivo", *SEM2_COLS]
    if stage == "early":
        drop_cols.extend(SEM1_COLS)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL].copy()

    if drop_missing_target:
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    return X, y
