"""Small behavioral checks for custom models and temporal feature selection."""
import unittest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from src.knn_propio import knn, minkowski_distance
from src.regresion_lineal_propia import LinearRegressor
from src.preprocess_clasificacion import get_classification_dataset
from src.preprocess_regresion import get_regression_dataset
from src.validacion_cruzada_propia import cross_validation_supervised


class TrainOnlyTransformer(TransformerMixin, BaseEstimator):
    """Assert that validation rows never appeared during fit."""
    seen_sizes = []
    def fit(self, X, y=None):
        self.seen_ = set(X.index)
        self.seen_sizes.append(len(X))
        self.fitting_ = True
        return self
    def transform(self, X):
        if self.fitting_:
            self.fitting_ = False
        else:
            assert self.seen_.isdisjoint(X.index)
        return X.to_numpy()


class CoreTests(unittest.TestCase):
    def test_knn_matches_reference_without_distance_ties(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(40, 4))
        y = np.array(['a', 'b'])[np.arange(40) % 2]
        query = rng.normal(size=(9, 4))
        for p in (1, 2, 3):
            model = knn()
            model.fit(X, y, k=5, p=p)
            expected = KNeighborsClassifier(n_neighbors=5, p=p).fit(X, y)
            np.testing.assert_array_equal(model.predict(query), expected.predict(query))
            np.testing.assert_allclose(model.predict_proba(query), expected.predict_proba(query))
            np.testing.assert_allclose(model.compute_distances(query[0]), [minkowski_distance(row, query[0], p) for row in X])
        with self.assertRaises(ValueError):
            model.fit(X, y, k=41)

    def test_linear_predictions_match_reference(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(60, 3))
        y = 2 + X @ np.array([1.5, -2, 3])
        expected = LinearRegression().fit(X, y).predict(X)
        np.testing.assert_allclose(LinearRegressor().fit(X, y).predict(X), expected, atol=1e-10)
        np.testing.assert_allclose(LinearRegressor().fit(X, y, method='gradient_descent', iterations=5000).predict(X), expected, atol=1e-6)

    def test_future_features_and_missing_targets(self):
        df = pd.DataFrame({'edad': [20, 21], 'nota_media_1sem': [12, 13],
                           'nota_media_2sem': [14, np.nan], 'objetivo': ['graduado', 'abandono']})
        X, _ = get_classification_dataset(df, stage='early')
        self.assertEqual(list(X.columns), ['edad'])
        X, y = get_regression_dataset(df)
        self.assertNotIn('objetivo', X)
        self.assertNotIn('nota_media_2sem', X)
        self.assertEqual(len(y), 1)
        self.assertEqual(y.iloc[0], 14)

    def test_cv_preprocessor_is_fit_only_on_each_training_fold(self):
        X = pd.DataFrame({'x': np.arange(30, dtype=float)})
        y = pd.Series(2 * X.x + 1)
        TrainOnlyTransformer.seen_sizes = []
        folds, _ = cross_validation_supervised(
            model_factory=lambda: LinearRegressor(), X=X, y=y,
            preprocessor=TrainOnlyTransformer(), n_folds=3, task='regression',
            scorers={'mae': lambda a,b: np.mean(np.abs(a-b))})
        self.assertEqual(TrainOnlyTransformer.seen_sizes, [20, 20, 20])
        self.assertLess(folds.mae.max(), 1e-10)

if __name__ == '__main__':
    unittest.main()
