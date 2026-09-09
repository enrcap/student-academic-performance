import numpy as np


class LogisticRegressorBinario:
    """Regresion logistica binaria entrenada con descenso por gradiente."""

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        penalty: str | None = None,
        l1_ratio: float = 0.5,
        C: float = 1.0,
        verbose: bool = False,
        print_every: int = 100,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud.")

        m, n = X.shape
        self.weights = np.zeros(n, dtype=float)
        self.bias = 0.0

        for i in range(num_iterations):
            y_hat = self.predict_proba(X)
            loss = self.log_likelihood(y, y_hat)

            if verbose and i % print_every == 0:
                print(f"Iteration {i}: Loss {loss}")

            error = y - y_hat
            dw = -(1 / m) * np.dot(X.T, error)
            db = -(1 / m) * np.sum(error)

            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def lasso_regularization(self, dw: np.ndarray, m: int, C: float) -> np.ndarray:
        return dw + (C / m) * np.sign(self.weights)

    def ridge_regularization(self, dw: np.ndarray, m: int, C: float) -> np.ndarray:
        return dw + (C / m) * self.weights

    def elasticnet_regularization(
        self,
        dw: np.ndarray,
        m: int,
        C: float,
        l1_ratio: float,
    ) -> np.ndarray:
        return (
            dw
            + l1_ratio * (C / m) * np.sign(self.weights)
            + (1 - l1_ratio) * (C / m) * self.weights
        )

    @staticmethod
    def log_likelihood(y: np.ndarray, y_hat: np.ndarray) -> float:
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
        m = y.shape[0]
        return -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


class LogisticRegressorOVR:
    """
    Regresion logistica multiclase mediante one-vs-rest.

    Entrena un clasificador binario por cada clase y predice usando la
    probabilidad mas alta entre los modelos binarios.
    """

    def __init__(self):
        self.classes_ = None
        self.models_ = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        penalty: str | None = None,
        l1_ratio: float = 0.5,
        C: float = 1.0,
        verbose: bool = False,
        print_every: int = 100,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.models_ = {}

        for class_label in self.classes_:
            binary_y = (y == class_label).astype(int)
            model = LogisticRegressorBinario()
            model.fit(
                X,
                binary_y,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                penalty=penalty,
                l1_ratio=l1_ratio,
                C=C,
                verbose=verbose,
                print_every=print_every,
            )
            self.models_[class_label] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probas = np.column_stack(
            [self.models_[class_label].predict_proba(X) for class_label in self.classes_]
        )

        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probas / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]

    def __str__(self):
        n_classes = 0 if self.classes_ is None else len(self.classes_)
        return f"LogisticRegressorOVR(n_classes={n_classes})"
