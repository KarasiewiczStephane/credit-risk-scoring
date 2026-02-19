"""Model training with LightGBM and Optuna hyperparameter optimization.

Includes baseline models (Logistic Regression, Random Forest) for comparison.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Train and optimize credit risk models.

    Supports LightGBM with Optuna tuning and baseline models for comparison.

    Attributes:
        random_state: Random seed for reproducibility.
        best_params: Best hyperparameters found by Optuna.
        models: Dictionary of trained models.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.best_params: dict[str, Any] = {}
        self.models: dict[str, Any] = {}

    def _objective(self, trial: optuna.Trial, x: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for LightGBM hyperparameter search.

        Args:
            trial: Optuna trial object.
            x: Training features.
            y: Training labels.

        Returns:
            Mean cross-validated AUC-ROC score.
        """
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, x, y, cv=3, scoring="roc_auc")
        return scores.mean()

    def train_lightgbm(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 30,
        timeout: int | None = 600,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM with Optuna hyperparameter tuning.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels.
            n_trials: Number of Optuna trials.
            timeout: Maximum optimization time in seconds.

        Returns:
            Trained LightGBM classifier with best hyperparameters.
        """
        logger.info("Starting Optuna optimization with %d trials", n_trials)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, x_train, y_train),
            n_trials=n_trials,
            timeout=timeout,
        )

        self.best_params["lightgbm"] = study.best_params
        logger.info("Best AUC-ROC: %.4f", study.best_value)
        logger.info("Best params: %s", study.best_params)

        best_model = lgb.LGBMClassifier(
            **study.best_params,
            objective="binary",
            random_state=self.random_state,
            verbosity=-1,
        )
        best_model.fit(x_train, y_train)
        self.models["lightgbm"] = best_model

        return best_model

    def train_baselines(self, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
        """Train baseline models for comparison.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            Dictionary of trained baseline models.
        """
        logger.info("Training baseline models")

        baselines = {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000, class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight="balanced",
            ),
        }

        for name, model in baselines.items():
            model.fit(x_train, y_train)
            self.models[name] = model
            logger.info("Trained %s", name)

        return baselines
