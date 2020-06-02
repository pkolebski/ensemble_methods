from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from data.load_dataset import load_dataset
from data import DATASETS_PATH


iris_x, iris_y = load_dataset(DATASETS_PATH / 'iris' / 'iris.data')
iris_dataset = ('Iris Dataset', iris_x, iris_y)
glass_x, glass_y = load_dataset(DATASETS_PATH / 'glass' / 'glass.data')
glass_dataset = ('Glass Dataset', glass_x, glass_y)
wine_x, wine_y = load_dataset(DATASETS_PATH / 'wine' / 'wine.data')
wine_dataset = ('Wine Dataset', wine_x, wine_y)
seeds_x, seeds_y = load_dataset(DATASETS_PATH / 'seeds' / 'seeds_dataset.csv', separator='\t')
seeds_dataset = ('Seeds dataset', seeds_x, seeds_y)


def test_bagging(base_estimator: BaseEstimator, features: np.ndarray, labels: np.ndarray,
                 cv_size: int, n_estimators: int, random_state: int = 42) -> float:
    bagging = BaggingClassifier(
        base_estimator,
        n_estimators=n_estimators,
        random_state=random_state
    )
    cv_score = np.mean(
        cross_val_score(
            bagging,
            features,
            labels,
            cv=cv_size,
        ),
    )
    return cv_score


def test_boosting(base_estimator: BaseEstimator, features: np.ndarray, labels: np.ndarray,
                 cv_size: int, n_estimators: int, lr: float = 0.001, random_state: int = 42) -> float:
    boosting = AdaBoostClassifier(
        base_estimator,
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    cv_score = np.mean(
        cross_val_score(
            boosting,
            features,
            labels,
            cv=cv_size,
        ),
    )
    return cv_score


def test_random_forest(features: np.ndarray, labels: np.ndarray, cv_size: int, n_estimators: int,
                       random_state: int = 42) -> np.ndarray:
    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    cv_score = np.mean(
        cross_val_score(
            random_forest,
            features,
            labels,
            cv=cv_size,
        ),
    )
    return cv_score


base_clf = DecisionTreeClassifier()

n_estimators = 200
cv_splits = 2
print(f"bagging: {test_bagging(base_clf, seeds_x, seeds_y, cv_splits, n_estimators):.2f}")
print(f"boosting: {test_boosting(base_clf, seeds_x, seeds_y, cv_splits, n_estimators):.2f}")
print(f"random forest: {test_random_forest(seeds_x, seeds_y, cv_splits, n_estimators):.2f}")
