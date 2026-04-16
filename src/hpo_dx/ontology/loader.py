from pathlib import Path
import pandas as pd


def load_hpo_terms(file_path: Path) -> pd.DataFrame:
    """Load the HPO catalog and prepare user-facing labels for the UI."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"HPO file not found: {file_path}. Please add the file or update the configured path."
        )

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(
            "Unsupported HPO file format. Please provide a .csv or .tsv file."
        )

    required_columns = {"HPO_CODE", "hpo_name"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The HPO file must contain at least the following columns: "
            f"{', '.join(sorted(required_columns))}. Missing: {', '.join(sorted(missing_columns))}."
        )

    df = df[["HPO_CODE", "hpo_name"]].dropna().drop_duplicates().copy()
    df = df.rename(columns={"HPO_CODE": "hpo_id"})

    # User-facing label, e.g. "HP:0001369 - Arthritis"
    df["display_label"] = df["hpo_id"] + " - " + df["hpo_name"]
    df = df.sort_values(["hpo_name", "hpo_id"]).reset_index(drop=True)
    return df


def load_ancestors_map(file_path: Path) -> dict[str, set[str]]:
    """Load the precomputed HPO ancestors map from a TSV or CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Ancestors map file not found: {file_path}. Please add the file or update the configured path."
        )

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(
            "Unsupported ancestors map format. Please provide a .csv or .tsv file."
        )

    required_columns = {"hpo_id", "ancestor_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The ancestors map file must contain at least the following columns: "
            f"{', '.join(sorted(required_columns))}. Missing: {', '.join(sorted(missing_columns))}."
        )

    df = df[["hpo_id", "ancestor_id"]].dropna().drop_duplicates().copy()

    ancestors_map = (
        df.groupby("hpo_id")["ancestor_id"]
        .apply(lambda values: set(values))
        .to_dict()
    )

    return ancestors_map