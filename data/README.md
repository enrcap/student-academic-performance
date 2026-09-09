# Datos y procedencia

`rendimiento_estudiantes.csv` contiene 4.424 filas, 36 predictores y la variable `objetivo`. El separador es `;` y la codificación es UTF-8.

## Fuente

La versión en español fue proporcionada en la asignatura de Aprendizaje Automático de ICAI. Se ha contrastado con **Predict Students' Dropout and Academic Success**, de UCI Machine Learning Repository:

- [Ficha del dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [DOI: 10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- Autores: Valentim Realinho, Mónica Vieira Martins, Jorge Machado y Luís Baptista (2021).
- Licencia del dataset original: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

El CSV distribuido aquí tiene nombres de columnas en español y sustituye códigos categóricos por etiquetas descriptivas. La comprobación fila a fila confirmó igualdad numérica y correspondencia biunívoca entre códigos y etiquetas para las 4.424 observaciones. `category_mapping.json` documenta esa correspondencia; `verification.json` recoge las columnas y el SHA-256 del archivo recibido. No se atribuye al autor del proyecto la recogida del dataset ni la traducción distribuida por la asignatura.

## Contenido

| Bloque | Ejemplos |
| --- | --- |
| Acceso y contexto | Curso, nota de admisión, edad al matricularse |
| Situación socioeconómica y administrativa | Cualificaciones y ocupaciones parentales, beca, matrícula al día |
| Primer semestre | Asignaturas matriculadas, evaluadas y aprobadas; nota media |
| Segundo semestre | Las mismas medidas del segundo semestre |
| Contexto macroeconómico | Desempleo, inflación y PIB |
| Etiqueta | `abandono`, `matriculado`, `graduado` |

Las etiquetas corresponden a la situación al final de la duración normal del curso, según la ficha de UCI. `matriculado` no significa necesariamente fracaso posterior.

## Tratamiento en el proyecto

El archivo incluido conserva los datos recibidos. Los módulos de preprocesado convierten `en_blanco` en faltante, conservan `desconocido` como categoría en el escenario principal y consideran no disponible una nota cero con cero evaluaciones. Esta última decisión es una hipótesis del proyecto, no una modificación de la definición oficial del dataset.

Para regresión se excluyen filas sin objetivo interpretable tras ese tratamiento. Las variables del segundo semestre se excluyen de los predictores supervisados. Los escenarios `early` también excluyen el primer semestre.
