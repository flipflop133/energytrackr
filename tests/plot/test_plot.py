import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from plot.plot import (
    prepare_commit_statistics,
    compute_distribution_and_normality,
    detect_change_points,
    plot_energy_data,
    create_energy_plot,
    create_energy_plots,
    EnergyPlotData,
)


@pytest.fixture
def sample_dataframe():
    data = {
        "commit": ["c1", "c1", "c2", "c2", "c3", "c3", "c3"],
        "energy-pkg": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 3.2],
        "energy-core": [0.9, 1.0, 1.8, 2.2, 3.0, 2.9, 3.1],
        "energy-gpu": [0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3],
    }
    return pd.DataFrame(data)


def test_prepare_commit_statistics(sample_dataframe):
    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(sample_dataframe, "energy-pkg")
    assert valid_commits == ["c1", "c2", "c3"]
    assert short_hashes == ["c1", "c2", "c3"]
    assert len(x_indices) == 3
    assert all(isinstance(x, float) for x in y_medians)
    assert all(isinstance(x, float) for x in y_errors)


def test_compute_distribution_and_normality(sample_dataframe):
    valid_commits = ["c1", "c2", "c3"]
    dist_data, normal_flags = compute_distribution_and_normality(sample_dataframe, valid_commits, "energy-core")
    assert len(dist_data) == 3
    assert len(normal_flags) == 3
    assert all(isinstance(x, (bool, np.bool_)) for x in normal_flags)


def test_detect_change_points():
    y = [1, 1, 1, 10, 10, 10]
    result = detect_change_points(y)
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


def test_plot_energy_data_runs(sample_dataframe):
    import matplotlib.pyplot as plt

    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(sample_dataframe, "energy-pkg")
    dist_data, normal_flags = compute_distribution_and_normality(sample_dataframe, valid_commits, "energy-pkg")
    cps = detect_change_points(y_medians)

    plot_data = EnergyPlotData(
        x_indices=x_indices,
        short_hashes=short_hashes,
        y_medians=y_medians,
        y_errors=y_errors,
        distribution_data=dist_data,
        normality_flags=normal_flags,
        change_points=cps,
        energy_column="energy-pkg",
    )

    fig, ax = plt.subplots()
    plot_energy_data(ax, plot_data)
    plt.close(fig)


def test_create_energy_plot_outputs_image(sample_dataframe, tmp_path: Path):
    file = tmp_path / "out.png"
    create_energy_plot(sample_dataframe, "energy-pkg", str(file))
    assert file.exists()
    assert file.stat().st_size > 0


def test_create_energy_plots(tmp_path: Path):
    df = pd.DataFrame(
        {
            "commit": ["abc123"] * 3 + ["def456"] * 3,
            "energy-pkg": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3],
            "energy-core": [0.9, 1.0, 1.1, 2.0, 2.1, 2.2],
            "energy-gpu": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )
    file = tmp_path / "myproject.csv"
    df.to_csv(file, header=False, index=False)

    create_energy_plots(str(file))

    images = list(tmp_path.glob("*_*_*.png"))
    assert len(images) == 3  # One per energy type
    assert all(f.stat().st_size > 0 for f in images)


import pytest
from plot.plot import create_energy_plots


def test_create_energy_plots_file_missing(monkeypatch):
    # Patch sys.exit directly (on the real sys module)
    monkeypatch.setattr(sys, "exit", lambda code=1: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit):
        create_energy_plots("nonexistent_file.csv")


from plot.plot import plot_energy_data, EnergyPlotData
import matplotlib.pyplot as plt
import numpy as np


def test_plot_energy_data_change_points_label():
    fig, ax = plt.subplots()

    plot_data = EnergyPlotData(
        x_indices=np.array([0, 1, 2]),
        short_hashes=["abc1234", "def5678", "ghi9012"],
        y_medians=[1.0, 5.0, 10.0],
        y_errors=[0.1, 0.2, 0.3],
        distribution_data=[np.array([1.0]), np.array([5.0]), np.array([10.0])],
        normality_flags=[True, True, True],
        change_points=[1, 2],  # 🔥 this ensures loop runs at least once
        energy_column="energy-pkg",
    )

    plot_energy_data(ax, plot_data)
    plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np
from plot.plot import EnergyPlotData, plot_energy_data


def test_plot_energy_data_multiple_change_points():
    fig, ax = plt.subplots()

    # Setup: 4 commits, 3 change points → loop runs 2x
    plot_data = EnergyPlotData(
        x_indices=np.array([0, 1, 2, 3]),
        short_hashes=["c0ffee0", "deadbee", "abc1234", "bada55"],
        y_medians=[1.0, 5.0, 2.0, 6.0],
        y_errors=[0.1, 0.2, 0.3, 0.2],
        distribution_data=[np.array([1.0])] * 4,
        normality_flags=[True, False, True, False],
        change_points=[1, 2, 4],  # ← .[:-1] makes it [1, 2], so loop runs twice
        energy_column="energy-pkg",
    )

    plot_energy_data(ax, plot_data)

    # Check labels added only once
    handles, labels = ax.get_legend_handles_labels()
    assert labels.count("Regression Start") == 1
    assert labels.count("Breaking Commit") == 1

    plt.close(fig)
