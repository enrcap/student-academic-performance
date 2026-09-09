# Memoria académica y versión del repositorio

[Informe académico original](Informe_academico_original.pdf) es la memoria entregada para la asignatura en el curso 2025/2026. Se conserva sin modificaciones como contexto del trabajo.

La versión preparada para GitHub revisa aspectos del protocolo de evaluación:

- Compara escenarios temporales mediante validación cruzada en entrenamiento.
- Evalúa la sensibilidad al tratamiento de valores faltantes dentro de entrenamiento.
- Selecciona el modelo principal por validación cruzada antes de evaluar el test.
- Corrige la posición de las barras de error en la figura de validación de clasificación.
- Vectoriza el cálculo de distancias de kNN sin cambiar la distancia de Minkowski ni la regla de voto.
- Publica notebooks ejecutados, tablas de resultados, dependencias comprobadas y atribución del dataset.

Las tablas de `../results/` y los notebooks representan la versión actual. La memoria original puede mostrar otras comparaciones o conclusiones; no se ha reescrito para hacerlas pasar por resultados del nuevo protocolo.

## Límites de la evaluación

Se conserva la partición aleatoria con semilla 42 ya explorada en la entrega original. Separar mejor selección y evaluación en el código no convierte ese test en una muestra nunca observada. Los resultados son retrospectivos: una evaluación confirmatoria necesita datos nuevos o cohortes posteriores.

Los escenarios excluyen explícitamente variables académicas futuras, pero no hay marcas de tiempo individuales que certifiquen cuándo se registraron los estados administrativos. La disponibilidad de variables como deuda o matrícula al día debe comprobarse antes de aplicar el modelo a una predicción real al ingreso.

La muestra procede de una institución concreta. No se ha validado su generalización a ICAI, otras universidades o cohortes nuevas. Los coeficientes indican asociaciones; los clusters describen estructura en una representación determinada por escalado, codificación y PCA. Ni los coeficientes ni los clusters establecen causalidad.
