# builtin_data_transforms/filter_outliers.py

"""Filter outliers from the DataFrame using IQR method."""

import pandas as pd

from energytrackr.plot.core.context import Context
from energytrackr.plot.core.interfaces import Transform


class FilterOutliers(Transform):
    """Filter outliers from the DataFrame using IQR method.

    Applies IQR filtering on a given energy column,
    writing filtered DataFrame back to ctx.artefacts['df'].
    """

    def __init__(self, column: str, multiplier: float = 1.5) -> None:
        """Initialize the FilterOutliers transform.

        Args:
            column (str): The column name to filter outliers from.
            multiplier (float): The multiplier for the IQR method. Default is 1.5.
        """
        self.column = column
        self.multiplier = multiplier

    def apply(self, ctx: Context) -> None:
        """Apply the FilterOutliers transform to the context.

        Filters outliers from the DataFrame using the IQR method and updates the context with the filtered DataFrame.

        Args:
            ctx (Context): The context containing the DataFrame and other artefacts.
        """
        df: pd.DataFrame = ctx.artefacts["df"]
        q1 = df[self.column].quantile(0.25)
        q3 = df[self.column].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - self.multiplier * iqr, q3 + self.multiplier * iqr
        df_f = df[(df[self.column] >= low) & (df[self.column] <= high)].copy()
        ctx.artefacts["df"] = df_f
