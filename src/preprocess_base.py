import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "rendimiento_estudiantes.csv"


def cargar_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset con el separador correcto."""
    return pd.read_csv(path, sep=";")


def permutar_filas(
    df: pd.DataFrame,
    random_state: int = 42,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Permuta aleatoriamente las filas de forma reproducible."""
    copia = df.sample(frac=1, random_state=random_state)
    if reset_index:
        copia = copia.reset_index(drop=True)
    return copia


def limpieza_basica(df: pd.DataFrame, desconocido_es_nan: bool = False) -> pd.DataFrame:
    """Limpieza comun para todas las tareas."""
    copia = df.copy()
    replacements = {"en_blanco": pd.NA}
    if desconocido_es_nan:
        replacements["desconocido"] = pd.NA
    copia = copia.replace(replacements)
    return copia


def tratar_ceros_notas_sin_evaluacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a valor faltante las notas medias iguales a 0 cuando el numero
    de asignaturas evaluadas en ese semestre tambien es 0.
    """
    copia = df.copy()

    pares = [
        ("nota_media_1sem", "asignaturas_1sem_evaluadas"),
        ("nota_media_2sem", "asignaturas_2sem_evaluadas"),
    ]

    for col_nota, col_eval in pares:
        mask = (copia[col_nota] == 0) & (copia[col_eval] == 0)
        copia.loc[mask, col_nota] = pd.NA

    return copia


def separa_tipos(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve columnas numericas y categoricas."""
    col_numericas = df.select_dtypes(include="number").columns.tolist()
    col_categoricas = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return col_numericas, col_categoricas
