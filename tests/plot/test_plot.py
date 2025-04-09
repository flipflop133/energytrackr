"""Unit tests for the plot module."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot.plot import (
    EnergyPlotData,
    compute_distribution_and_normality,
    create_energy_plot,
    create_energy_plots,
    detect_change_points,
    plot_energy_data,
    prepare_commit_statistics,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to create a sample DataFrame for testing."""
    data = {
        "commit": ["c1", "c1", "c2", "c2", "c3", "c3", "c3"],
        "energy-pkg": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 3.2],
        "energy-core": [0.9, 1.0, 1.8, 2.2, 3.0, 2.9, 3.1],
        "energy-gpu": [0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3],
    }
    return pd.DataFrame(data)


def test_prepare_commit_statistics(sample_dataframe: pd.DataFrame) -> None:
    """Test the preparation of commit statistics."""
    valid_commits, short_hashes, x_indices, y_medians, y_errors = prepare_commit_statistics(sample_dataframe, "energy-pkg")
    assert valid_commits == ["c1", "c2", "c3"]
    assert short_hashes == ["c1", "c2", "c3"]
    expected_x_indices_count = 3
    assert len(x_indices) == expected_x_indices_count
    assert all(isinstance(x, float) for x in y_medians)
    assert all(isinstance(x, float) for x in y_errors)


def test_compute_distribution_and_normality(sample_dataframe: pd.DataFrame) -> None:
    """Test the computation of distribution and normality."""
    valid_commits = ["c1", "c2", "c3"]
    dist_data, normal_flags = compute_distribution_and_normality(sample_dataframe, valid_commits, "energy-core")
    expected_dist_data_count = 3
    assert len(dist_data) == expected_dist_data_count
    expected_normal_flags_count = 3
    assert len(normal_flags) == expected_normal_flags_count
    assert all(isinstance(x, bool | np.bool_) for x in normal_flags)


def test_detect_change_points() -> None:
    """Test the detection of change points."""
    y = [1, 1, 1, 10, 10, 10]
    result = detect_change_points(y)
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


def test_plot_energy_data_runs(sample_dataframe: pd.DataFrame) -> None:
    """Test the plotting of energy data."""
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


def test_create_energy_plot_outputs_image(sample_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Test the creation of an energy plot and check if the output file is created."""
    file = tmp_path / "out.png"
    create_energy_plot(sample_dataframe, "energy-pkg", str(file))
    assert file.exists()
    assert file.stat().st_size > 0


def test_create_energy_plots(tmp_path: Path) -> None:
    """Test the creation of multiple energy plots and check if the output files are created."""
    df = pd.DataFrame(
        {
            "commit": ["abc123"] * 3 + ["def456"] * 3,
            "energy-pkg": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3],
            "energy-core": [0.9, 1.0, 1.1, 2.0, 2.1, 2.2],
            "energy-gpu": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
    )
    file = tmp_path / "myproject.csv"
    df.to_csv(file, header=False, index=False)

    create_energy_plots(str(file))

    images = list(tmp_path.glob("*_*_*.png"))
    expected_images_count = len(df.columns) - 1  # One image per energy type
    assert len(images) == expected_images_count
    assert all(f.stat().st_size > 0 for f in images)


def test_create_energy_plots_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the behavior when the input file is missing."""
    # Patch sys.exit directly (on the real sys module)
    monkeypatch.setattr(sys, "exit", lambda code=1: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit):
        create_energy_plots("nonexistent_file.csv")


def test_plot_energy_data_change_points_label() -> None:
    """Test the plotting of energy data with change points and check labels."""
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


def test_plot_energy_data_multiple_change_points() -> None:
    """Test the plotting of energy data with multiple change points."""
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
