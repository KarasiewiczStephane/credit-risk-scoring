"""Before/after fairness comparison report generation.

Generates comparison tables and comprehensive reports for fairness
mitigation evaluation.
"""

from typing import Any

import pandas as pd

from src.fairness.analyzer import FairnessMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FairnessReportGenerator:
    """Generate before/after fairness comparison reports.

    Attributes:
        protected_attributes: List of protected attribute names.
    """

    def __init__(self, protected_attributes: list[str]) -> None:
        self.protected_attributes = protected_attributes

    def generate_comparison_report(
        self,
        before: dict[str, FairnessMetrics],
        after: dict[str, FairnessMetrics],
        mitigation_method: str,
    ) -> pd.DataFrame:
        """Generate before/after comparison table.

        Args:
            before: Fairness metrics before mitigation.
            after: Fairness metrics after mitigation.
            mitigation_method: Name of the mitigation method applied.

        Returns:
            DataFrame with before/after metric comparison.
        """
        rows = []
        for attr in self.protected_attributes:
            if attr not in before or attr not in after:
                continue

            b = before[attr]
            a = after[attr]

            for metric_name, b_val, a_val in [
                ("demographic_parity_diff", b.demographic_parity_diff, a.demographic_parity_diff),
                ("equalized_odds_diff", b.equalized_odds_diff, a.equalized_odds_diff),
            ]:
                improvement_pct = (1 - abs(a_val) / abs(b_val)) * 100 if b_val != 0 else 0
                rows.append(
                    {
                        "attribute": attr,
                        "mitigation_method": mitigation_method,
                        "metric": metric_name,
                        "before": b_val,
                        "after": a_val,
                        "improvement": b_val - a_val,
                        "improvement_pct": improvement_pct,
                    }
                )

        return pd.DataFrame(rows)

    def generate_full_report(
        self,
        before: dict[str, FairnessMetrics],
        after_by_method: dict[str, dict[str, FairnessMetrics]],
    ) -> dict[str, Any]:
        """Generate comprehensive fairness report.

        Args:
            before: Fairness metrics before mitigation.
            after_by_method: Fairness metrics after each mitigation method.

        Returns:
            Dictionary with summary, metrics, and recommendations.
        """
        report: dict[str, Any] = {
            "summary": {
                "protected_attributes": self.protected_attributes,
                "mitigation_methods_evaluated": list(after_by_method.keys()),
            },
            "before_mitigation": {
                attr: {
                    "demographic_parity_diff": m.demographic_parity_diff,
                    "equalized_odds_diff": m.equalized_odds_diff,
                    "overall_accuracy": m.overall_accuracy,
                }
                for attr, m in before.items()
            },
            "after_mitigation": {},
            "recommendations": [],
        }

        best_method = None
        best_improvement = 0.0

        for method, after in after_by_method.items():
            report["after_mitigation"][method] = {
                attr: {
                    "demographic_parity_diff": m.demographic_parity_diff,
                    "equalized_odds_diff": m.equalized_odds_diff,
                    "overall_accuracy": m.overall_accuracy,
                }
                for attr, m in after.items()
            }

            total_improvement = 0.0
            for attr in self.protected_attributes:
                if attr in before and attr in after:
                    total_improvement += abs(before[attr].demographic_parity_diff) - abs(
                        after[attr].demographic_parity_diff
                    )

            if total_improvement > best_improvement:
                best_improvement = total_improvement
                best_method = method

        if best_method:
            report["recommendations"].append(f"Best mitigation method: {best_method}")
            report["recommendations"].append(
                f"Average fairness improvement: {best_improvement:.4f}"
            )

        logger.info("Generated fairness report with %d methods", len(after_by_method))
        return report
