"""Tests for model training and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data for model testing."""
    x, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
        weights=[0.7, 0.3],
    )
    split = 240
    return x[:split], x[split:], y[:split], y[split:]


@pytest.fixture
def trainer() -> ModelTrainer:
    """Create a model trainer instance."""
    return ModelTrainer(random_state=42)


@pytest.fixture
def evaluator() -> ModelEvaluator:
    """Create a model evaluator instance."""
    return ModelEvaluator()


class TestModelTrainer:
    """Tests for ModelTrainer."""

    def test_train_baselines(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
    ) -> None:
        x_train, _, y_train, _ = training_data
        baselines = trainer.train_baselines(x_train, y_train)
        assert "logistic_regression" in baselines
        assert "random_forest" in baselines
        assert "logistic_regression" in trainer.models
        assert "random_forest" in trainer.models

    def test_baselines_can_predict(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
    ) -> None:
        x_train, x_test, y_train, _ = training_data
        trainer.train_baselines(x_train, y_train)
        for model in trainer.models.values():
            preds = model.predict(x_test)
            assert len(preds) == len(x_test)
            assert set(preds).issubset({0, 1})

    def test_train_lightgbm_quick(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
    ) -> None:
        x_train, x_test, y_train, _ = training_data
        model = trainer.train_lightgbm(x_train, y_train, n_trials=3, timeout=60)
        preds = model.predict(x_test)
        assert len(preds) == len(x_test)
        assert "lightgbm" in trainer.best_params
        assert "lightgbm" in trainer.models

    def test_optuna_stores_best_params(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
    ) -> None:
        x_train, _, y_train, _ = training_data
        trainer.train_lightgbm(x_train, y_train, n_trials=3, timeout=60)
        params = trainer.best_params["lightgbm"]
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "max_depth" in params


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_evaluate_returns_all_metrics(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
    ) -> None:
        x_train, x_test, y_train, y_test = training_data
        trainer.train_baselines(x_train, y_train)
        model = trainer.models["logistic_regression"]
        metrics = evaluator.evaluate(model, x_test, y_test)
        assert "auc_roc" in metrics
        assert "gini" in metrics
        assert "ks_statistic" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_auc_in_valid_range(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
    ) -> None:
        x_train, x_test, y_train, y_test = training_data
        trainer.train_baselines(x_train, y_train)
        metrics = evaluator.evaluate(trainer.models["logistic_regression"], x_test, y_test)
        assert 0 <= metrics["auc_roc"] <= 1

    def test_gini_formula(self) -> None:
        assert ModelEvaluator.calculate_gini(0.5) == 0.0
        assert ModelEvaluator.calculate_gini(1.0) == 1.0
        assert ModelEvaluator.calculate_gini(0.75) == pytest.approx(0.5)

    def test_ks_statistic_in_range(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = ModelEvaluator.calculate_ks_statistic(y_true, y_prob)
        assert 0 <= ks <= 1

    def test_compare_models(
        self,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
    ) -> None:
        x_train, x_test, y_train, y_test = training_data
        trainer.train_baselines(x_train, y_train)
        comparison = evaluator.compare_models(trainer.models, x_test, y_test)
        assert isinstance(comparison, pd.DataFrame)
        assert comparison.index.name == "model"
        assert len(comparison) == 2
        assert "auc_roc" in comparison.columns

    def test_perfect_classifier_metrics(self, evaluator: ModelEvaluator) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ks = ModelEvaluator.calculate_ks_statistic(y_true, y_prob)
        assert ks == pytest.approx(1.0)

    def test_random_classifier_gini(self) -> None:
        gini = ModelEvaluator.calculate_gini(0.5)
        assert gini == pytest.approx(0.0)
