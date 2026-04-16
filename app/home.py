from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import json
# import joblib
import xgboost as xgb


APP_TITLE = "hpo-dx"
APP_SUBTITLE = (
    "Phenotype-driven differential diagnosis support for recurrent fever syndromes"
)

# Resolve paths relative to the repository root so the app remains reproducible
# after cloning, regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPO_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_terms_expanded_with_ancestors.csv"
DEFAULT_ANCESTORS_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_ancestors_map.tsv"
DEFAULT_FEATURES_FILE = REPO_ROOT / "assets" / "model" / "baked_feature_names.txt"
DEFAULT_PREPROCESSING_FILE = REPO_ROOT / "assets" / "model" / "preprocessing_metadata.json"
DEFAULT_MODEL_BUNDLE = REPO_ROOT / "assets" / "model" / "ovr_bundle.joblib"
DEFAULT_MODEL_DIR = REPO_ROOT / "assets" / "model"


# These should match the levels expected by the training preprocessing.
GENDER_OPTIONS = ["Female", "Male"]
ETHNICITY_OPTIONS = [
    # "Ethnicity group 2",
    # "Ethnicity group 4",
    # "Other ethnicity",
    "0: African, African descent",
    "1: Arctic (Siberian, Eskimo)",
    "2: Caucasian (European)",
    "3: Caucasian (Indian)",
    "4: Caucasian (Middle East)",
    "5: Caucasian (North African, Other)",
    "6: Hispanic (for US)",
    "7: Indigenous Australian",
    "8: Mixed Ethnicity",
    "9: Native American",
    "10: North East Asian (Mongol, Tibetan, Korean Japanese, etc)",
    "11: Pacific (Polynesian, Micronesian, etc)",
    "12: South East Asian (Chinese, Thai, Malay, Filipino, etc)",
    "13: Other Ethnicity",
    "14: Cannot be reported"
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


def load_feature_names(file_path: Path) -> list[str]:
    """Load the final baked feature names used by the trained model."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {file_path}. Please add the file or update the configured path."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    if not feature_names:
        raise ValueError(f"No feature names found in {file_path}.")

    return feature_names


def load_preprocessing_metadata(file_path: Path) -> dict:
    """Load preprocessing metadata required to build the model input vector."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Preprocessing metadata file not found: {file_path}. Please add the file or update the configured path."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


# def load_model_bundle(file_path: Path) -> dict:
#     """Load the serialized OVR model bundle."""
#     if not file_path.exists():
#         raise FileNotFoundError(
#             f"Model bundle not found: {file_path}. Please add the file or update the configured path."
#         )

#     bundle = joblib.load(file_path)

#     if "models" not in bundle:
#         raise ValueError("Invalid model bundle: missing 'models' key.")

#     return bundle


def load_xgb_models_from_json(model_dir: Path) -> dict:
    """Load OVR XGBoost models saved as JSON."""
    disease_files = {
        "CAPS": model_dir / "CAPS.json",
        "FMF": model_dir / "FMF.json",
        "MKD": model_dir / "MKD.json",
        "PFAPA": model_dir / "PFAPA.json",
        "TRAPS": model_dir / "TRAPS.json",
    }

    models = {}

    for disease, path in disease_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

        model = xgb.XGBClassifier()
        model.load_model(path)

        models[disease] = model

    return models


def build_model_bundle_from_json(model_dir: Path) -> dict:
    models = load_xgb_models_from_json(model_dir)

    bundle = {
        "models": models,
        "class_names": list(models.keys()),
        # opzionale: aggiungi weights se li hai
        # "f2_weights_train_cv": {...}
    }

    return bundle


# --- helpers ---
def scale_numeric(value: float, mean_value: float, sd_value: float) -> float:
    """Apply training-based standardization to a numeric variable."""
    if sd_value == 0:
        return 0.0
    return (value - mean_value) / sd_value


def encode_gender(gender_value: str | None) -> int:
    """Encode gender into the baked dummy representation."""
    return int(gender_value == "Male")


def encode_ethnicity(ethnicity_value: str | None) -> tuple[int, int]:
    """Encode ethnicity into the baked dummy representation."""
    # Training encoding:
    # code 2 -> ethnicity_X4 = 0, ethnicity_other = 0
    # code 4 -> ethnicity_X4 = 1, ethnicity_other = 0
    # all other codes -> ethnicity_X4 = 0, ethnicity_other = 1
    if ethnicity_value == "2: Caucasian (European)":
        return 0, 0

    if ethnicity_value == "4: Caucasian (Middle East)":
        return 1, 0

    return 0, 1
# ---------------


def build_feature_vector(
    feature_names: list[str],
    preprocessing_metadata: dict,
    active_hpo_ids: set[str],
    age_value: float,
    fever_length_value: float,
    gender_value: str,
    ethnicity_value: str,
) -> pd.DataFrame:
    """Build a single-row feature vector aligned to the trained model."""
    row = {feature_name: 0.0 for feature_name in feature_names}

    # Encode active HPO terms using the baked naming convention
    for hpo_id in active_hpo_ids:
        baked_hpo_feature = f"{hpo_id}_X1"
        if baked_hpo_feature in row:
            row[baked_hpo_feature] = 1.0

    # Scale numeric variables
    if "age" in row:
        row["age"] = scale_numeric(
            age_value,
            preprocessing_metadata["age_mean"],
            preprocessing_metadata["age_sd"],
        )

    if "fever_length" in row:
        row["fever_length"] = scale_numeric(
            fever_length_value,
            preprocessing_metadata["fever_length_mean"],
            preprocessing_metadata["fever_length_sd"],
        )

    # Encode gender
    gender_column = preprocessing_metadata.get("gender_dummy_column", "gender_Male")
    if gender_column in row:
        row[gender_column] = float(encode_gender(gender_value))

    # Encode ethnicity
    ethnicity_x4, ethnicity_other = encode_ethnicity(ethnicity_value)

    if "ethnicity_X4" in row:
        row["ethnicity_X4"] = float(ethnicity_x4)

    if "ethnicity_other" in row:
        row["ethnicity_other"] = float(ethnicity_other)

    return pd.DataFrame([row], columns=feature_names)


def predict_ovr_probabilities(
    model_bundle: dict,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Generate raw and normalized OVR probabilities for a single patient."""
    models = model_bundle["models"]
    class_names = model_bundle.get("class_names", list(models.keys()))
    weights = model_bundle.get("f2_weights_train_cv")

    raw_scores: dict[str, float] = {}

    for disease in class_names:
        model = models[disease]
        raw_scores[disease] = float(model.predict_proba(feature_frame)[0, 1])

    raw_sum = sum(raw_scores.values())
    if raw_sum == 0:
        norm_scores = {disease: 0.0 for disease in class_names}
    else:
        norm_scores = {
            disease: raw_scores[disease] / raw_sum
            for disease in class_names
        }

    results_df = pd.DataFrame(
        {
            "disease": class_names,
            # "raw_probability": [raw_scores[disease] for disease in class_names], # for debugging
            "normalized_probability": [norm_scores[disease] for disease in class_names],
        }
    )

    if weights is not None:
        weighted_raw = {
            disease: raw_scores[disease] * float(weights[disease])
            for disease in class_names
        }
        weighted_sum = sum(weighted_raw.values())
        if weighted_sum == 0:
            weighted_scores = {disease: 0.0 for disease in class_names}
        else:
            weighted_scores = {
                disease: weighted_raw[disease] / weighted_sum
                for disease in class_names
            }

        results_df["weighted_probability"] = [
            weighted_scores[disease] for disease in class_names
        ]

    results_df = results_df.sort_values(
        "normalized_probability",
        ascending=False,
    ).reset_index(drop=True)

    return results_df


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
    st.session_state.age_value = None
    st.session_state.fever_length_value = None
    st.session_state.gender_value = None
    st.session_state.ethnicity_value = None


def request_prediction() -> None:
    """Mark that the user has requested model inference."""
    st.session_state.prediction_requested = True


def render_selected_terms(selected_ids: list[str], hpo_df: pd.DataFrame) -> None:
    """Render the currently selected HPO terms in insertion order."""
    st.markdown("##### Selected HPO terms")

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
    # st.markdown("### Clinical input")
    st.write(f"- **Age:** {st.session_state.age_value}")
    st.write(f"- **Fever length (days):** {st.session_state.fever_length_value}")
    st.write(f"- **Gender:** {st.session_state.gender_value}")
    st.write(f"- **Ethnicity:** {st.session_state.ethnicity_value}")


def render_prediction_results(results_df: pd.DataFrame) -> None:
    """Render prediction probabilities and a sorted bar chart."""
    st.markdown("### Prediction output")

    top_disease = results_df.iloc[0]["disease"]
    top_probability = results_df.iloc[0]["normalized_probability"]

    st.success(f"Most probable disease: {top_disease} ({top_probability:.1%})")

    display_df = results_df.copy()
    # display_df["raw_probability"] = display_df["raw_probability"].round(3)
    display_df["normalized_probability"] = display_df["normalized_probability"].round(3)

    if "weighted_probability" in display_df.columns:
        display_df["weighted_probability"] = display_df["weighted_probability"].round(3)

    st.dataframe(display_df, width="stretch", hide_index=True)

    chart_df = results_df[["disease", "normalized_probability"]].copy()
    chart_df = chart_df.set_index("disease")

    st.markdown("### Probability bar plot")
    st.bar_chart(chart_df)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        # page_icon="🏥",
        page_icon=":material/assistant_device:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()

    try:
        feature_names = load_feature_names(DEFAULT_FEATURES_FILE)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    try:
        preprocessing_metadata = load_preprocessing_metadata(DEFAULT_PREPROCESSING_FILE)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    # try:
    #     model_bundle = load_model_bundle(DEFAULT_MODEL_BUNDLE)
    # except Exception as exc:
    #     st.error(str(exc))
    #     st.stop()


    try:
        model_bundle = build_model_bundle_from_json(DEFAULT_MODEL_DIR)
    except Exception as exc:
        st.error(str(exc))
        st.stop()


    initialize_session_state()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:


        with st.container():
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
                    "Age at onset",
                    min_value=0.0,
                    step=0.1,
                    value=st.session_state.age_value,
                    placeholder="Enter age at onset...",
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

        # st.divider()
        st.space(size="xxsmall")

        with st.container():

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
                    width='stretch',
                    disabled=selected_label is None,
                    on_click=add_hpo,
                    args=(selected_label, label_to_id),
                )

            with predict_col:
                st.button(
                    "Generate prediction",
                    type="primary",
                    width='stretch',
                    on_click=request_prediction,
                    disabled=not prediction_ready,
                )

            with undo_col:
                st.button(
                    "Undo last",
                    width='stretch',
                    disabled=not st.session_state.selected_hpo_ids,
                    on_click=remove_last_hpo,
                )

            with reset_col:
                st.button(
                    "Reset all",
                    width='stretch',
                    disabled=not st.session_state.selected_hpo_ids
                    and st.session_state.age_value is None
                    and st.session_state.fever_length_value is None
                    and st.session_state.gender_value is None
                    and st.session_state.ethnicity_value is None,
                    on_click=reset_all,
                )

        st.divider()
        st.markdown("### Entered variables:")
        render_clinical_summary()
        # st.divider()
        st.space(size="xxsmall")
        render_selected_terms(st.session_state.selected_hpo_ids, hpo_df)





    with right_col:

        st.markdown("## Prediction results")
        st.caption("Probabilistic output across the five diseases.")

        if st.session_state.prediction_requested:
            feature_vector = build_feature_vector(
                feature_names=feature_names,
                preprocessing_metadata=preprocessing_metadata,
                active_hpo_ids=st.session_state.active_hpo_ids,
                age_value=st.session_state.age_value,
                fever_length_value=st.session_state.fever_length_value,
                gender_value=st.session_state.gender_value,
                ethnicity_value=st.session_state.ethnicity_value,
            )

            results_df = predict_ovr_probabilities(
                model_bundle=model_bundle,
                feature_frame=feature_vector,
            )

            render_prediction_results(results_df)
        else:
            st.info("Prediction results will appear here after all required inputs are provided.")



if __name__ == "__main__":
    main()