"""Weight of Evidence (WoE) transformation and Information Value (IV) calculation.

Implements optimal binning for continuous features and WoE/IV computation
for credit risk feature analysis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WoEBin:
    """Container for a single WoE bin's statistics."""

    lower: float
    upper: float
    woe: float
    iv: float
    event_rate: float
    count: int


class WoETransformer:
    """Weight of Evidence transformer with optimal binning.

    Performs WoE transformation on features using decision-tree-based
    optimal binning for continuous features and direct calculation for
    categorical features.

    Attributes:
        min_bins: Minimum number of bins for continuous features.
        max_bins: Maximum number of bins for continuous features.
        min_bin_size: Minimum fraction of data per bin.
        woe_dict: WoE bins per feature after fitting.
        iv_values: Information Value per feature after fitting.
    """

    def __init__(self, min_bins: int = 3, max_bins: int = 10, min_bin_size: float = 0.05) -> None:
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.woe_dict: dict[str, list[WoEBin]] = {}
        self.iv_values: dict[str, float] = {}
        self._bin_edges: dict[str, list[float]] = {}
        self._is_continuous: dict[str, bool] = {}

    def _calculate_woe_iv(
        self, df: pd.DataFrame, feature: str, target: str
    ) -> tuple[list[WoEBin], float]:
        """Calculate WoE and IV for a binned feature.

        Args:
            df: DataFrame containing the feature and target columns.
            feature: Name of the feature column.
            target: Name of the target column.

        Returns:
            Tuple of (list of WoEBin, total IV).
        """
        total_events = df[target].sum()
        total_non_events = len(df) - total_events

        bins = []
        total_iv = 0.0

        for value in sorted(df[feature].unique()):
            mask = df[feature] == value
            events = df.loc[mask, target].sum()
            non_events = mask.sum() - events

            # Laplace smoothing to avoid log(0)
            dist_events = (events + 0.5) / (total_events + 1)
            dist_non_events = (non_events + 0.5) / (total_non_events + 1)

            woe = np.log(dist_non_events / dist_events)
            iv = (dist_non_events - dist_events) * woe

            bins.append(
                WoEBin(
                    lower=value,
                    upper=value,
                    woe=float(woe),
                    iv=float(iv),
                    event_rate=float(events / mask.sum()) if mask.sum() > 0 else 0.0,
                    count=int(mask.sum()),
                )
            )
            total_iv += iv

        return bins, float(total_iv)

    def _optimal_binning(self, series: pd.Series, target: pd.Series) -> pd.Series:
        """Apply optimal binning using decision tree splits.

        Args:
            series: Feature values to bin.
            target: Binary target values.

        Returns:
            Series with bin labels assigned to each value.
        """
        n_samples = len(series)
        min_samples = max(int(n_samples * self.min_bin_size), 2)

        dt = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins, min_samples_leaf=min_samples, random_state=42
        )
        dt.fit(series.values.reshape(-1, 1), target)

        thresholds = sorted(dt.tree_.threshold[dt.tree_.feature == 0])
        thresholds = [series.min() - 1] + thresholds + [series.max() + 1]

        return pd.cut(series, bins=thresholds, labels=False, duplicates="drop")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        """Fit WoE transformer on training data.

        Args:
            x: Feature DataFrame.
            y: Binary target Series.

        Returns:
            self
        """
        df = x.copy()
        df["target"] = y.values

        logger.info("Fitting WoE transformer on %d features", len(x.columns))

        for col in x.columns:
            if x[col].nunique() > self.max_bins:
                self._is_continuous[col] = True
                binned = self._optimal_binning(x[col], y)
                df[f"{col}_binned"] = binned

                # Store bin edges for transform
                thresholds = sorted(
                    DecisionTreeClassifier(
                        max_leaf_nodes=self.max_bins,
                        min_samples_leaf=max(int(len(x) * self.min_bin_size), 2),
                        random_state=42,
                    )
                    .fit(x[col].values.reshape(-1, 1), y)
                    .tree_.threshold[
                        DecisionTreeClassifier(
                            max_leaf_nodes=self.max_bins,
                            min_samples_leaf=max(int(len(x) * self.min_bin_size), 2),
                            random_state=42,
                        )
                        .fit(x[col].values.reshape(-1, 1), y)
                        .tree_.feature
                        == 0
                    ]
                )
                self._bin_edges[col] = [x[col].min() - 1] + thresholds + [x[col].max() + 1]

                bins, iv = self._calculate_woe_iv(df, f"{col}_binned", "target")
            else:
                self._is_continuous[col] = False
                bins, iv = self._calculate_woe_iv(df, col, "target")

            self.woe_dict[col] = bins
            self.iv_values[col] = iv

        logger.info("WoE fitting complete. Top IV features computed.")
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WoE values.

        Args:
            x: Feature DataFrame to transform.

        Returns:
            DataFrame with WoE-encoded features.
        """
        x_woe = x.copy()
        for col in x.columns:
            if col in self.woe_dict:
                if self._is_continuous.get(col, False) and col in self._bin_edges:
                    binned = pd.cut(
                        x[col], bins=self._bin_edges[col], labels=False, duplicates="drop"
                    )
                    woe_map = {b.lower: b.woe for b in self.woe_dict[col]}
                    x_woe[col] = binned.map(woe_map).fillna(0)
                else:
                    woe_map = {b.lower: b.woe for b in self.woe_dict[col]}
                    x_woe[col] = x[col].map(woe_map).fillna(0)
        return x_woe

    def get_iv_report(self) -> pd.DataFrame:
        """Get Information Value report for all features.

        Returns:
            DataFrame with feature names, IV values, and predictive power.
        """
        iv_df = pd.DataFrame(
            [
                {"feature": k, "iv": v, "predictive_power": self._iv_strength(v)}
                for k, v in self.iv_values.items()
            ]
        ).sort_values("iv", ascending=False)
        return iv_df.reset_index(drop=True)

    @staticmethod
    def _iv_strength(iv: float) -> str:
        """Classify IV value into predictive power categories.

        Args:
            iv: Information Value.

        Returns:
            String describing the predictive power.
        """
        if iv < 0.02:
            return "Not useful"
        elif iv < 0.1:
            return "Weak"
        elif iv < 0.3:
            return "Medium"
        elif iv < 0.5:
            return "Strong"
        else:
            return "Suspicious (check overfitting)"
