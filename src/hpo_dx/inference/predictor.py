import pandas as pd



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