import pandas as pd


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


def get_unsupervised_dataset(df: pd.DataFrame, profile: str = "entry") -> pd.DataFrame:
    """
    profile='entry' -> solo variables de entrada y contexto
    profile='sem1'  -> incluye informacion del primer semestre
    """
    if profile == "entry":
        drop_cols = ["objetivo", *SEM1_COLS, *SEM2_COLS]
    elif profile == "sem1":
        drop_cols = ["objetivo", *SEM2_COLS]
    else:
        raise ValueError("profile debe ser 'entry' o 'sem1'")

    return df.drop(columns=drop_cols, errors="ignore")
