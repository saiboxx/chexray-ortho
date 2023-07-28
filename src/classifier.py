"""A collection of classifier settings."""
from typing import Optional, Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from xgboost import XGBClassifier

from src.net import NNClassifier


def get_logistic_regression(
    penalty: Optional[str] = None,
    solver: str = 'saga',
    max_iter: int = 100,
    verbose: bool = 1,
    n_jobs: int = -1,
    **kwargs,
) -> LogisticRegression:
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        verbose=verbose,
        n_jobs=n_jobs,
        **kwargs,
    )


def get_linear_nn(max_epochs: int = 3, batch_size: int = 256) -> NNClassifier:
    return NNClassifier(max_epochs=max_epochs, batch_size=batch_size)


def get_non_linear_nn(max_epochs: int = 3, batch_size: int = 256) -> NNClassifier:
    model = nn.Sequential(
        nn.LazyLinear(512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return NNClassifier(model=model, max_epochs=max_epochs, batch_size=batch_size)


def get_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    class_weight: Any = 'balanced',
    verbose: int = 1,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        verbose=verbose,
        n_jobs=n_jobs,
        class_weight=class_weight**kwargs,
    )


def get_xgboost(
    silent: bool = False,
    n_jobs: int = -1,
) -> XGBClassifier:
    return XGBClassifier(n_jobs=n_jobs, silent=silent)


def get_classifier(name: str, classifier_args: Optional[Dict] = None) -> Any:
    if classifier_args is None:
        classifier_args = {}

    match name:
        case 'log_reg':
            func = get_logistic_regression
        case 'nn':
            func = get_linear_nn
        case 'nn_nl':
            func = get_non_linear_nn
        case 'rf':
            func = get_random_forest
        case 'xgb':
            func = get_xgboost
        case _:
            raise ValueError(f'Classifier "{name}" unknown.')
    return func(**classifier_args)
