import streamlit as st
import pandas as pd


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