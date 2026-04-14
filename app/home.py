from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


APP_TITLE = "hpo-dx"
APP_SUBTITLE = (
    "Phenotype-driven differential diagnosis support for recurrent fever syndromes"
)

# Resolve paths relative to the repository root so the app remains reproducible
# after cloning, regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPO_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_terms_expanded_with_ancestors.csv"
DEFAULT_ANCESTORS_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_ancestors_map.tsv"

# These should match the levels expected by the training preprocessing.
GENDER_OPTIONS = ["Female", "Male"]
# ETHNICITY_OPTIONS = ["2", "4", "0", "3", "5", "6", "8", "10", "12", "13", "14"]
ETHNICITY_OPTIONS = [
    "Ethnicity group 2",
    "Ethnicity group 4",
    "Other ethnicity",
]


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


def initialize_session_state() -> None:
    """Initialize session state used to store clinical and HPO inputs."""
    if "selected_hpo_ids" not in st.session_state:
        st.session_state.selected_hpo_ids = []

    if "selected_hpo_label" not in st.session_state:
        st.session_state.selected_hpo_label = None

    if "active_hpo_ids" not in st.session_state:
        st.session_state.active_hpo_ids = set()

    if "prediction_requested" not in st.session_state:
        st.session_state.prediction_requested = False

    if "ancestors_map" not in st.session_state:
        st.session_state.ancestors_map = {}

    if "age_value" not in st.session_state:
        st.session_state.age_value = None

    if "fever_length_value" not in st.session_state:
        st.session_state.fever_length_value = None

    if "gender_value" not in st.session_state:
        st.session_state.gender_value = None

    if "ethnicity_value" not in st.session_state:
        st.session_state.ethnicity_value = None


def expand_with_ancestors(
    selected_ids: list[str], ancestors_map: dict[str, set[str]]
) -> set[str]:
    """Return the full active HPO set after ancestor back-propagation."""
    active_ids = set(selected_ids)

    for hpo_id in selected_ids:
        active_ids.update(ancestors_map.get(hpo_id, set()))

    return active_ids


def add_hpo(selected_label: str, label_to_id: dict[str, str]) -> None:
    """Add one HPO term to the current selection if it is not already active."""
    selected_id = label_to_id[selected_label]

    if selected_id in st.session_state.active_hpo_ids:
        st.warning(
            "This HPO term is already active in the current profile, either as a selected term or as an inferred ancestor."
        )
        st.session_state.selected_hpo_label = None
        return

    if selected_id not in st.session_state.selected_hpo_ids:
        st.session_state.selected_hpo_ids.append(selected_id)

    st.session_state.active_hpo_ids = expand_with_ancestors(
        st.session_state.selected_hpo_ids,
        st.session_state.ancestors_map,
    )
    st.session_state.prediction_requested = False
    st.session_state.selected_hpo_label = None


def remove_last_hpo() -> None:
    """Remove the last selected HPO term, if present, and recompute active terms."""
    if st.session_state.selected_hpo_ids:
        st.session_state.selected_hpo_ids.pop()

    st.session_state.active_hpo_ids = expand_with_ancestors(
        st.session_state.selected_hpo_ids,
        st.session_state.ancestors_map,
    )
    st.session_state.prediction_requested = False


def reset_all() -> None:
    """Clear the full clinical and HPO input state."""
    st.session_state.selected_hpo_ids = []
    st.session_state.selected_hpo_label = None
    st.session_state.active_hpo_ids = set()
    st.session_state.prediction_requested = False
    st.session_state.age_value = 0.0
    st.session_state.fever_length_value = 0.0
    st.session_state.gender_value = GENDER_OPTIONS[0]
    st.session_state.ethnicity_value = ETHNICITY_OPTIONS[0]


def request_prediction() -> None:
    """Mark that the user has requested model inference."""
    st.session_state.prediction_requested = True


def render_selected_terms(selected_ids: list[str], hpo_df: pd.DataFrame) -> None:
    """Render the currently selected HPO terms in insertion order."""
    st.markdown("### Selected HPO terms")

    if not selected_ids:
        st.info("No HPO terms selected yet.")
        return

    selected_df = hpo_df[hpo_df["hpo_id"].isin(selected_ids)].copy()
    order_map = {hpo_id: idx for idx, hpo_id in enumerate(selected_ids)}
    selected_df["sort_order"] = selected_df["hpo_id"].map(order_map)
    selected_df = selected_df.sort_values("sort_order")

    for _, row in selected_df.iterrows():
        st.write(f"- **{row['hpo_id']}** - {row['hpo_name']}")


def render_clinical_summary() -> None:
    """Render the current clinical input summary."""
    st.markdown("### Clinical input")
    st.write(f"- **Age:** {st.session_state.age_value}")
    st.write(f"- **Fever length (days):** {st.session_state.fever_length_value}")
    st.write(f"- **Gender:** {st.session_state.gender_value}")
    st.write(f"- **Ethnicity:** {st.session_state.ethnicity_value}")


def render_prediction_placeholder(selected_ids: list[str]) -> None:
    """Temporary placeholder for the future model prediction output."""
    st.markdown("### Prediction output")

    if not selected_ids:
        st.warning("Please select at least one HPO term before generating a prediction.")
        return

    st.success("Input captured correctly. Model inference will be connected in the next step.")

    st.write("Current clinical input:")
    st.code(
        "\n".join(
            [
                f"age = {st.session_state.age_value}",
                f"fever_length = {st.session_state.fever_length_value}",
                f"gender = {st.session_state.gender_value}",
                f"ethnicity = {st.session_state.ethnicity_value}",
            ]
        ),
        language="text",
    )

    st.write("Current user-selected HPO input:")
    st.code("\n".join(selected_ids), language="text")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":medical_symbol:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()

    try:
        hpo_df = load_hpo_terms(DEFAULT_HPO_FILE)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    try:
        ancestors_map = load_ancestors_map(DEFAULT_ANCESTORS_FILE)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    valid_hpo_ids = set(hpo_df["hpo_id"])

    # Keep only source HPO terms present in the current HPO catalog,
    # and keep only ancestors that are also part of the same valid HPO space.
    ancestors_map = {
        hpo_id: {ancestor for ancestor in ancestors if ancestor in valid_hpo_ids}
        for hpo_id, ancestors in ancestors_map.items()
        if hpo_id in valid_hpo_ids
    }

    initialize_session_state()
    st.session_state.ancestors_map = ancestors_map

    label_to_id = dict(zip(hpo_df["display_label"], hpo_df["hpo_id"]))
    available_labels = hpo_df["display_label"].tolist()

    clinical_input_complete = all(
        value is not None
        for value in [
            st.session_state.age_value,
            st.session_state.fever_length_value,
            st.session_state.gender_value,
            st.session_state.ethnicity_value,
        ]
    )

    prediction_ready = (
        clinical_input_complete
        and bool(st.session_state.selected_hpo_ids)
    )

    st.markdown("### Clinical variables")

    clinical_col1, clinical_col2 = st.columns(2)

    with clinical_col1:
        st.number_input(
            "Age",
            min_value=0.0,
            step=0.1,
            value=st.session_state.age_value,
            placeholder="Enter age...",
            key="age_value",
        )

        st.selectbox(
            "Gender",
            options=GENDER_OPTIONS,
            index=None,
            placeholder="Select gender...",
            key="gender_value",
        )

    with clinical_col2:
        st.number_input(
            "Fever length (days)",
            min_value=0.0,
            step=0.1,
            value=st.session_state.fever_length_value,
            placeholder="Enter fever length...",
            key="fever_length_value",
        )

        st.selectbox(
            "Ethnicity",
            options=ETHNICITY_OPTIONS,
            index=None,
            placeholder="Select ethnicity group...",
            key="ethnicity_value",
        )

    st.divider()

    st.markdown("### Insert HPO terms")
    selected_label = st.selectbox(
        "Search and select one HPO term at a time",
        options=available_labels,
        index=None,
        placeholder="Start typing a symptom name or an HPO code...",
        key="selected_hpo_label",
    )

    add_col, predict_col, undo_col, reset_col = st.columns([1, 1, 1, 1])

    with add_col:
        st.button(
            "Add HPO",
            use_container_width=True,
            disabled=selected_label is None,
            on_click=add_hpo,
            args=(selected_label, label_to_id),
        )

    with predict_col:
        st.button(
            "Generate prediction",
            type="primary",
            use_container_width=True,
            on_click=request_prediction,
            disabled=not prediction_ready,
        )

    with undo_col:
        st.button(
            "Undo last",
            use_container_width=True,
            disabled=not st.session_state.selected_hpo_ids,
            on_click=remove_last_hpo,
        )

    with reset_col:
        st.button(
            "Reset all",
            use_container_width=True,
            disabled=not st.session_state.selected_hpo_ids
            and st.session_state.age_value == 0.0
            and st.session_state.fever_length_value == 0.0
            and st.session_state.gender_value == GENDER_OPTIONS[0]
            and st.session_state.ethnicity_value == ETHNICITY_OPTIONS[0],
            on_click=reset_all,
        )

    st.divider()
    render_clinical_summary()
    st.divider()
    render_selected_terms(st.session_state.selected_hpo_ids, hpo_df)

    if st.session_state.prediction_requested:
        st.divider()
        render_prediction_placeholder(st.session_state.selected_hpo_ids)


if __name__ == "__main__":
    main()