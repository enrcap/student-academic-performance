import numpy as np


class LinearRegressor:
    """
    Regresion lineal propia basada en las practicas p2-3 y p2-4.

    Soporta ajuste por ecuacion normal y por descenso por gradiente.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def fit_simple(self, X, y):
        """Ajusta una regresion lineal simple con una sola variable."""
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.coefficients = np.cov(X, y, ddof=0)[0, 1] / np.var(X, ddof=0)
        self.intercept = np.mean(y) - self.coefficients * np.mean(X)
        return self

    def fit_multiple(self, X, y):
        """
        Ajusta la regresion lineal multiple mediante ecuacion normal.

        Se usa pseudoinversa para mayor estabilidad numerica con datos reales.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_bias = np.c_[np.ones(X.shape[0]), X]
        beta = np.linalg.pinv(X_bias) @ y

        self.intercept = float(beta[0])
        self.coefficients = beta[1:]
        return self

    def fit_gradient_descent(
        self,
        X,
        y,
        learning_rate: float = 0.01,
        iterations: int = 5000,
    ):
        """Ajusta la regresion lineal multiple mediante descenso por gradiente."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_bias = np.c_[np.ones(X.shape[0]), X]
        m = X_bias.shape[0]
        theta = np.zeros(X_bias.shape[1], dtype=float)
        self.loss_history = []

        for epoch in range(iterations):
            predictions = X_bias @ theta
            error = predictions - y
            gradient = (X_bias.T @ error) / m
            theta -= learning_rate * gradient

            if epoch % 100 == 0 or epoch == iterations - 1:
                mse = float(np.mean(error ** 2))
                self.loss_history.append((epoch, mse))

        self.intercept = float(theta[0])
        self.coefficients = theta[1:]
        return self

    def fit(
        self,
        X,
        y,
        method: str = "least_squares",
        learning_rate: float = 0.01,
        iterations: int = 5000,
    ):
        """Ajusta el modelo con ecuacion normal o descenso por gradiente."""
        if method == "least_squares":
            return self.fit_multiple(X, y)
        if method == "gradient_descent":
            return self.fit_gradient_descent(
                X,
                y,
                learning_rate=learning_rate,
                iterations=iterations,
            )
        raise ValueError(
            "method debe ser 'least_squares' o 'gradient_descent'."
        )

    def predict(self, X):
        """Predice valores continuos usando el modelo ajustado."""
        if self.coefficients is None or self.intercept is None:
            raise ValueError("El modelo aun no ha sido ajustado.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.intercept + X @ self.coefficients


def evaluate_regression(y_true, y_pred):
    """Calcula R2, RMSE y MAE para una tarea de regresion."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - rss / tss
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": float(r_squared), "RMSE": float(rmse), "MAE": float(mae)}
