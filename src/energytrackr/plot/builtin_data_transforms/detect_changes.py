"""Detect changes in distributions of energy data over time."""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import ttest_ind

from energytrackr.plot.config import get_settings
from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Configurable, Transform
from energytrackr.utils.logger import logger


@dataclass
class EffectSize:
    """Represents the effect size of a statistical comparison.

    Attributes:
        cohen_d (float): The calculated Cohen's d value indicating the standardized difference between two means.
        category (str): A qualitative description of the effect size (e.g., 'small', 'medium', 'large').
    """

    cohen_d: float
    category: str


@dataclass
class ChangeMagnitude:
    """Represents the magnitude of change between two values, including both percentage and absolute differences.

    Attributes:
        pct_change (float): The percentage change between two values.
        pct_change_level (str): A qualitative description of the percentage change (e.g., 'low', 'moderate', 'high').
        abs_diff (float): The absolute difference between two values.
        practical_level (str): A qualitative assessment of the practical significance of the change.
    """

    pct_change: float
    pct_change_level: str
    abs_diff: float
    practical_level: str


@dataclass
class ChangeEvent:
    """Represents a change event with associated metadata."""

    index: int
    direction: str
    p_value: float
    effect_size: EffectSize
    change_magnitude: ChangeMagnitude
    context_tags: list[str] | None  # e.g. [“cpu”, “module:io”]
    level: int


@dataclass(frozen=True)
class DetectChangesConfig:
    """Configuration for detecting changes in distributions of energy measurements."""

    column: str | None = None
    thresholds: dict | None = None
    tags: list[str] | None = None  # e.g. [“cpu”, “module:io”]


class DetectChanges(Transform, Configurable[DetectChangesConfig]):
    """Detect changes in distributions of energy data over time."""

    def __init__(self, **params: dict[str, Any]) -> None:
        """Initialize the DetectChanges transform.

        Args:
            **params: Configuration parameters for the transform.
        """
        super().__init__(DetectChangesConfig, **params)
        self.column = self.config.column or None
        settings = get_settings()
        cfg = settings.energytrackr.analysis.thresholds
        incoming = self.config.thresholds or {}
        self.thr = {
            "welch_p": incoming.get("welch_p", cfg.welch_p),
            "min_pct_increase": incoming.get("min_pct_increase", cfg.min_pct_increase),
            "cohen_d": incoming.get("cohen_d", cfg.cohen_d),
            "pct_change": incoming.get("pct_change", cfg.pct_change),
            "practical": incoming.get("practical", cfg.practical),
            "min_values_for_normality_test": incoming.get(
                "min_values_for_normality_test",
                cfg.min_values_for_normality_test,
            ),
        }

    def apply(self, ctx: Context) -> None:
        """Simplified apply method using helper to process each pair.

        Args:
            ctx (Context): The context containing the DataFrame and other artefacts.
        """
        changes: list[ChangeEvent] = []
        distributions = ctx.artefacts.get("distributions", [])

        for idx in range(1, len(distributions)):
            baseline = distributions[idx - 1]
            test = distributions[idx]
            if event := self._process_pair(idx, baseline, test, ctx):
                changes.append(event)

        ctx.artefacts["change_events"] = changes

    def _process_pair(self, index: int, baseline: np.ndarray, test: np.ndarray, ctx: Context) -> ChangeEvent | None:
        """Analyze a pair of baseline and test samples to detect statistically significant changes.

        This method compares two numeric sample arrays (baseline and test) at a given index, performing statistical tests and
        effect size calculations to determine if a meaningful change has occurred.

        Args:
            index (int): The index corresponding to the pair of samples being analyzed.
            baseline (np.ndarray): The baseline sample data.
            test (np.ndarray): The test sample data.
            ctx (Context): The context containing the DataFrame and other artefacts.

        Returns:
            ChangeEvent | None: A ChangeEvent object describing the detected change if statistically significant, or None if
            no significant change is detected or if sample sizes are too small.

        The method performs the following steps:
            - Skips processing if either sample is too small for statistical testing.
            - Computes a p-value to test for significant difference between samples.
            - If the p-value is above the configured threshold, returns None.
            - Calculates Cohen's d effect size and classifies its magnitude.
            - Computes median values, percent change, and absolute difference between samples.
            - Classifies the percent change and practical significance.
            - Determines the direction and overall level of the change.
            - Returns a ChangeEvent encapsulating all relevant change information.
        """
        # Skip small samples
        if len(baseline) < self.thr["min_values_for_normality_test"] or len(test) < self.thr["min_values_for_normality_test"]:
            return None

        level = None
        if (p_value := self._compute_p_value(baseline, test)) >= self.thr["welch_p"]:
            level = 0

        cohen_d = self._compute_cohen_d(baseline, test)
        effect_cat = self.classify_effect_size(cohen_d)
        effect_size = EffectSize(cohen_d=cohen_d, category=effect_cat)

        baseline_med = float(np.median(baseline))
        test_med = float(np.median(test))

        pct = abs((test_med - baseline_med) / baseline_med)
        pct_cat = self.classify_pct_change(pct)

        abs_diff = test_med - baseline_med
        practical = self.classify_practical(abs_diff, baseline_med)
        change_magnitude = ChangeMagnitude(
            pct_change=pct,
            pct_change_level=pct_cat,
            abs_diff=abs_diff,
            practical_level=practical,
        )
        commit = ctx.stats.get("valid_commits", [])[index]
        if level is None:
            level = 5 if self.detect_level_5(ctx, commit) else self._determine_level(cohen_d, pct, practical)
        return ChangeEvent(
            index=index,
            direction=self._get_direction(baseline_med, test_med),
            p_value=p_value,
            effect_size=effect_size,
            change_magnitude=change_magnitude,
            context_tags=None,
            level=level,
        )

    @staticmethod
    def _compute_p_value(baseline: np.ndarray, test: np.ndarray) -> float:
        """Compute Welch's t-test p-value.

        Args:
            baseline (np.ndarray): The baseline sample data.
            test (np.ndarray): The test sample data.

        Returns:
            float: The p-value from the t-test.
        """
        _, p = ttest_ind(baseline, test, equal_var=False)
        return float(p)

    @staticmethod
    def _compute_cohen_d(baseline: np.ndarray, test: np.ndarray) -> float:
        """Compute Cohen's d effect size.

        Args:
            baseline (np.ndarray): The baseline sample data.
            test (np.ndarray): The test sample data.

        Returns:
            float: The Cohen's d effect size.
        """
        mean_b, mean_t = np.mean(baseline), np.mean(test)
        var_b = np.var(baseline, ddof=1)
        var_t = np.var(test, ddof=1)
        pooled = np.sqrt((var_b + var_t) / 2.0) or 1.0
        return float((mean_t - mean_b) / pooled)

    def _determine_level(self, cohen_d: float, pct: float, practical: str) -> int:
        """Determine change level based on thresholds.

        Args:
            cohen_d (float): The Cohen's d effect size.
            pct (float): The percent change.
            practical (str): The practical significance level.

        Returns:
            int: The change level (1-5).
        """
        practical_level_4_thresholds = [
            level for level, threshold in self.thr["practical"].items() if threshold > self.thr["practical"]["info"]
        ]
        level = 1
        if abs(cohen_d) >= self.thr["cohen_d"]["negligible"]:
            level = max(level, 2)
        if pct >= self.thr["min_pct_increase"]:
            level = max(level, 3)
        if practical in practical_level_4_thresholds:
            level = max(level, 4)
        # Context tags (future) would bump to 5
        return level

    def detect_level_5(self, ctx: Context, commit: str) -> bool:
        """Detect level 5 changes based on context tags.

        Args:
            ctx (Context): The context containing the DataFrame and other artefacts.
            commit (str): The commit hash to check for level 5 changes.

        Returns:
            bool: True if level 5 changes are detected, False otherwise.
        """
        if not self.config.tags:
            return False
        commit_msg = ctx.artefacts["commit_details"][commit].get("commit_message", "")
        if not commit_msg:
            logger.warning("No commit message found for commit %s", commit)
            return False
        tags = self.config.tags
        for tag in tags:
            # Match tag as a whole word, case-insensitive
            if re.search(rf"\b{re.escape(tag)}\b", commit_msg, re.IGNORECASE):
                logger.info("Detected level 5 change: tag '%s' found in commit message.", tag)
                return True
        return False

    def classify_effect_size(self, d: float) -> str:
        """Classify the effect size based on Cohen's d value.

        Args:
            d (float): The Cohen's d effect size.

        Returns:
            str: The category of the effect size.
        """
        abs_d = abs(d)
        for category, thresh in self.thr["cohen_d"].items():
            if abs_d <= thresh:
                return category
        return list(self.thr["cohen_d"].keys())[-1]

    def classify_pct_change(self, pct: float) -> str:
        """Classify the percent change based on predefined thresholds.

        Args:
            pct (float): The percent change value.

        Returns:
            str: The category of the percent change.
        """
        for label, threshold in self.thr["pct_change"].items():
            if pct < threshold:
                return label
        return list(self.thr["pct_change"].keys())[-1]

    def classify_practical(self, abs_diff: float, baseline_median: float) -> str:
        """Classify the practical significance based on absolute difference and baseline median.

        Args:
            abs_diff (float): The absolute difference between test and baseline medians.
            baseline_median (float): The median of the baseline sample.

        Returns:
            str: The category of practical significance.
        """
        thresholds = {level: factor * baseline_median for level, factor in self.thr["practical"].items()}
        for level, limit in sorted(thresholds.items(), key=lambda x: x[1]):
            if abs_diff < limit:
                return level
        return "critical"

    @staticmethod
    def _get_direction(baseline_med: float, test_med: float) -> str:
        return "increase" if test_med > baseline_med else "decrease"
