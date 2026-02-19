"""Traditional credit scorecard with WoE transformation and point allocation.

Implements a logistic-regression-based scorecard with PDO (Points to Double Odds)
scaling and score-to-PD mapping.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.woe_transformer import WoETransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScorecardConfig:
    """Configuration for credit scorecard scaling.

    Attributes:
        pdo: Points to Double the Odds.
        base_score: Score at the base odds.
        base_odds: Good-to-bad ratio at the base score.
    """

    pdo: int = 20
    base_score: int = 600
    base_odds: float = 50.0


@dataclass
class FeatureScorecard:
    """Scorecard details for a single feature.

    Attributes:
        feature: Feature name.
        bins: Bin labels.
        woe_values: WoE values per bin.
        points: Allocated points per bin.
        coefficient: Logistic regression coefficient for this feature.
    """

    feature: str
    bins: list[str]
    woe_values: list[float]
    points: list[int]
    coefficient: float


class CreditScorecard:
    """Traditional credit scorecard with WoE and point allocation.

    Uses WoE transformation followed by logistic regression to build
    a point-based scorecard with PDO scaling.

    Attributes:
        config: Scorecard configuration (PDO, base score, base odds).
        woe_transformer: Fitted WoE transformer.
        logreg: Fitted logistic regression model.
        factor: Scaling factor derived from PDO.
        offset: Offset derived from base score and base odds.
        feature_scorecards: Per-feature scorecard details.
    """

    def __init__(self, config: ScorecardConfig | None = None) -> None:
        self.config = config or ScorecardConfig()
        self.woe_transformer = WoETransformer()
        self.logreg = LogisticRegression(random_state=42, max_iter=1000)
        self.factor: float = 0.0
        self.offset: float = 0.0
        self.feature_scorecards: dict[str, FeatureScorecard] = {}

    def _calculate_scaling_factors(self) -> tuple[float, float]:
        """Calculate Factor and Offset for score scaling.

        Score = Offset + Factor * log(Odds)
        PDO = Factor * log(2)  =>  Factor = PDO / log(2)
        Base_Score = Offset + Factor * log(Base_Odds)

        Returns:
            Tuple of (factor, offset).
        """
        self.factor = self.config.pdo / np.log(2)
        self.offset = self.config.base_score - self.factor * np.log(self.config.base_odds)
        return self.factor, self.offset

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CreditScorecard":
        """Fit scorecard: WoE transform -> Logistic Regression -> Point allocation.

        Args:
            x: Training feature DataFrame.
            y: Training target Series.

        Returns:
            self
        """
        logger.info("Fitting credit scorecard on %d samples", len(x))

        # WoE transformation
        self.woe_transformer.fit(x, y)
        x_woe = self.woe_transformer.transform(x)

        # Logistic regression on WoE features
        self.logreg.fit(x_woe, y)

        # Scaling factors
        self._calculate_scaling_factors()

        # Point allocation per feature bin
        intercept = self.logreg.intercept_[0]
        coefficients = self.logreg.coef_[0]
        n_features = len(x.columns)
        intercept_contribution = intercept / n_features

        for i, col in enumerate(x.columns):
            coef = coefficients[i]
            woe_bins = self.woe_transformer.woe_dict.get(col, [])

            points = []
            bin_labels = []
            woe_values = []

            for bin_info in woe_bins:
                woe = bin_info.woe
                point = -int((woe * coef + intercept_contribution) * self.factor)
                points.append(point)
                bin_labels.append(str(bin_info.lower))
                woe_values.append(woe)

            self.feature_scorecards[col] = FeatureScorecard(
                feature=col,
                bins=bin_labels,
                woe_values=woe_values,
                points=points,
                coefficient=float(coef),
            )

        logger.info("Scorecard fitted with %d features", len(self.feature_scorecards))
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        """Calculate credit scores for given features.

        Args:
            x: Feature DataFrame to score.

        Returns:
            Array of integer credit scores.
        """
        x_woe = self.woe_transformer.transform(x)
        log_odds = self.logreg.decision_function(x_woe)
        scores = self.offset + self.factor * log_odds
        return scores.astype(int)

    def score_to_pd(self, scores: np.ndarray) -> np.ndarray:
        """Convert credit scores to Probability of Default.

        Args:
            scores: Array of credit scores.

        Returns:
            Array of default probabilities.
        """
        log_odds = (scores - self.offset) / self.factor
        odds = np.exp(log_odds)
        return 1 / (1 + odds)

    def get_scorecard_table(self) -> pd.DataFrame:
        """Generate scorecard table for interpretation.

        Returns:
            DataFrame with Feature, Bin, WoE, and Points columns.
        """
        rows = []
        for sc in self.feature_scorecards.values():
            for bin_label, woe, points in zip(sc.bins, sc.woe_values, sc.points, strict=False):
                rows.append(
                    {
                        "Feature": sc.feature,
                        "Bin": bin_label,
                        "WoE": round(woe, 4),
                        "Points": points,
                    }
                )
        return pd.DataFrame(rows)

    def explain_score(self, x_row: pd.Series) -> dict:
        """Explain score breakdown for a single application.

        Args:
            x_row: Feature values for one application.

        Returns:
            Dictionary with total_score, probability_of_default, and breakdown.
        """
        total_score = int(self.offset)
        breakdown = [{"component": "Base Score", "points": int(self.offset)}]

        for col in x_row.index:
            if col in self.feature_scorecards:
                sc = self.feature_scorecards[col]
                value = x_row[col]
                for i, bin_val in enumerate(sc.bins):
                    if str(value) == bin_val:
                        pts = sc.points[i]
                        total_score += pts
                        breakdown.append({"feature": col, "value": value, "points": pts})
                        break

        return {
            "total_score": total_score,
            "probability_of_default": float(self.score_to_pd(np.array([total_score]))[0]),
            "breakdown": breakdown,
        }
