import pandas as pd


TARGET_COL = "objetivo"

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


def get_classification_dataset(
    df: pd.DataFrame,
    stage: str = "sem1",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    stage='early'  -> sin variables de 1er ni 2o semestre
    stage='sem1'   -> con variables de 1er semestre y sin 2o semestre
    """
    if stage not in {"early", "sem1"}:
        raise ValueError("stage debe ser 'early' o 'sem1'")

    drop_cols = [TARGET_COL, *SEM2_COLS]
    if stage == "early":
        drop_cols.extend(SEM1_COLS)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL].copy()
    return X, y
