#!/usr/bin/env python3
"""Merge multiple energy-measurement CSVs into one.

Preserving input order and dropping duplicate commits. Supports files with or without headers.
"""

import argparse
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        An argparse.Namespace with:
          - inputs: list of input CSV file paths
          - output: path for the merged CSV
          - hash_column: name of the commit-hash column
          - data_column: name of the energy-value column
          - no_header: flag indicating input files lack headers
    """
    parser = argparse.ArgumentParser(
        description=("Merge multiple CSV files of energy measurements, dropping duplicate commit hashes."),
    )
    parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="+",
        help="Input CSV files (first is base; others are merged in order)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to write the merged CSV output",
    )
    parser.add_argument(
        "--hash-column",
        default="commit_hash",
        help="Name of the commit-hash column (default: 'commit_hash')",
    )
    parser.add_argument(
        "--data-column",
        default="energy",
        help="Name of the energy-value column (default: 'energy')",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat all input files as headerless two-column CSVs",
    )
    return parser.parse_args()


def merge_csv_files(input_files: list[str], hash_column: str, data_column: str, no_header: bool) -> pd.DataFrame:
    """Merge a list of CSVs into one DataFrame, dropping duplicate commit hashes.

    Args:
        input_files: List of paths to CSV files, in merge order.
        hash_column: Name for the commit-hash column.
        data_column: Name for the energy-value column.
        no_header: If True, treat every file as having no header and exactly
                   two columns (hash then value).

    Returns:
        A pandas.DataFrame with all unique commits in input order.
    """
    merged = pd.DataFrame()
    seen_hashes = set()

    for idx, path in enumerate(input_files, start=1):
        try:
            if no_header:
                df = pd.read_csv(path, header=None)
                if df.shape[1] < 2:
                    sys.exit(f"Error: '{path}' has fewer than 2 columns.")
                # Keep only first two columns
                df = df.iloc[:, :2]
                df.columns = [hash_column, data_column]
            else:
                df = pd.read_csv(path)
        except Exception as e:
            sys.exit(f"Error reading '{path}': {e}")

        if hash_column not in df.columns:
            sys.exit(f"Error: Column '{hash_column}' not found in '{path}'. Available columns: {list(df.columns)}")

        if idx == 1:
            # First file: take everything
            merged = df.copy()
            seen_hashes.update(merged[hash_column].astype(str).tolist())
        else:
            # Later files: only new hashes
            mask_new = ~df[hash_column].astype(str).isin(seen_hashes)
            new_rows = df.loc[mask_new]
            merged = pd.concat([merged, new_rows], ignore_index=True)
            seen_hashes.update(new_rows[hash_column].astype(str).tolist())

    return merged


def main() -> None:
    """Main entry point: parse args, merge CSVs, and write the output."""
    args = parse_args()

    if len(args.inputs) < 2:
        sys.exit("Please provide at least two input CSV files to merge.")

    merged_df = merge_csv_files(
        args.inputs,
        hash_column=args.hash_column,
        data_column=args.data_column,
        no_header=args.no_header,
    )

    try:
        merged_df.to_csv(args.output, index=False)
    except Exception as e:
        sys.exit(f"Error writing output '{args.output}': {e}")

    print(f"Merged {len(args.inputs)} files into '{args.output}' ({len(merged_df)} total unique commits).")


if __name__ == "__main__":
    main()
