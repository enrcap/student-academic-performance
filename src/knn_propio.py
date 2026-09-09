import numpy as np


def minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 2) -> float:
    """Calcula la distancia de Minkowski entre dos vectores."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


class knn:
    """Implementacion propia de kNN para clasificacion."""

    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Ajusta el modelo almacenando el conjunto de entrenamiento.

        Requisitos:
        - X_train e y_train deben tener el mismo numero de filas.
        - k y p deben ser enteros positivos.
        """
        if len(X_train) != len(y_train):
            raise ValueError("X_train e y_train deben tener la misma longitud.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k debe ser un entero positivo.")
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p debe ser un entero positivo.")

        if k > len(X_train):
            raise ValueError("k no puede superar el número de muestras de entrenamiento.")

        self.k = k
        self.p = p
        self.x_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        self.classes_ = np.unique(self.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice la clase para cada muestra de entrada."""
        predictions = []
        for point in np.asarray(X, dtype=float):
            distances = self.compute_distances(point)
            knn_indices = self.get_k_nearest_neighbors(distances)
            knn_labels = self.y_train[knn_indices]
            predictions.append(self.most_common_label(knn_labels))

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad estimada de cada clase."""
        probabilities = []

        for point in np.asarray(X, dtype=float):
            distances = self.compute_distances(point)
            indices = self.get_k_nearest_neighbors(distances)
            nearest_labels = self.y_train[indices]
            prob = np.array([np.sum(nearest_labels == c) / self.k for c in self.classes_])
            probabilities.append(prob)

        return np.array(probabilities)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Calcula la distancia entre un punto y todos los puntos de train."""
        return np.sum(np.abs(self.x_train - point) ** self.p, axis=1) ** (1 / self.p)

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Obtiene los indices de los k vecinos mas cercanos."""
        return np.argsort(distances)[: self.k]

    def most_common_label(self, knn_labels: np.ndarray):
        """Devuelve la etiqueta mas frecuente entre los vecinos cercanos."""
        values, counts = np.unique(knn_labels, return_counts=True)
        return values[np.argmax(counts)]

    def __str__(self):
        return f"kNN propio (k={self.k}, p={self.p})"
