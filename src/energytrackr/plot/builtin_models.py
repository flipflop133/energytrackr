"""Module for statistical analysis and change detection in energy consumption data.

This module defines data classes and constants used for statistical analysis and change detection in energy consumption data.
It includes classes for representing effect sizes, change magnitudes, and change events, as well as constants for thresholds
and settings related to statistical tests and practical significance.
"""

from dataclasses import dataclass

MIN_MEASUREMENTS = 2
NORMALITY_P_THRESHOLD = 0.05
MIN_VALUES_FOR_NORMALITY_TEST = 3
WELCH_P_THRESHOLD = 0.05
MIN_PCT_INCREASE = 0.02
COHEN_D_THRESHOLDS: dict[str, float] = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": 1.2,
}
PCT_CHANGE_THRESHOLDS: dict[str, float] = {
    "minor": 0.05,
    "moderate": 0.10,
    "major": float("inf"),
}

PRACTICAL_THRESHOLDS = {
    "info": 0.05,
    "warning": 0.1,
    "critical": 0.2,
}
PRACTICAL_LEVEL_4_TRIGGERS = [
    level for level, threshold in PRACTICAL_THRESHOLDS.items() if threshold > PRACTICAL_THRESHOLDS["info"]
]


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
    """Represents a change event with associated metadata.

    Attributes:
        index (int): The index of the change event.
        severity (float): The severity level of the change event.
        direction (str): The direction of the change event (e.g., "up", "down").
        cohen_d (float): The Cohen's d effect size associated with the change event.
    """

    index: int
    direction: str
    p_value: float
    effect_size: EffectSize
    change_magnitude: ChangeMagnitude
    context_tags: list[str] | None  # e.g. [“cpu”, “module:io”]
    level: int
