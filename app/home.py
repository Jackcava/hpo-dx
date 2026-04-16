from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpo_dx.ontology.loader import load_hpo_terms, load_ancestors_map
from hpo_dx.ontology.ancestors import expand_with_ancestors
from hpo_dx.inference.loader import load_feature_names, load_preprocessing_metadata, build_model_bundle_from_json
from hpo_dx.inference.feature_builder import build_feature_vector
from hpo_dx.inference.predictor import predict_ovr_probabilities
from hpo_dx.ui.rendering import *


APP_TITLE = "hpo-dx"
APP_SUBTITLE = (
    "Phenotype-driven differential diagnosis support for recurrent fever syndromes"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPO_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_terms_expanded_with_ancestors.csv"
DEFAULT_ANCESTORS_FILE = REPO_ROOT / "assets" / "ontology" / "hpo_ancestors_map.tsv"
DEFAULT_FEATURES_FILE = REPO_ROOT / "assets" / "model" / "baked_feature_names.txt"
DEFAULT_PREPROCESSING_FILE = REPO_ROOT / "assets" / "model" / "preprocessing_metadata.json"
DEFAULT_MODEL_DIR = REPO_ROOT / "assets" / "model"


GENDER_OPTIONS = ["Female", "Male"]

ETHNICITY_OPTIONS = [
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