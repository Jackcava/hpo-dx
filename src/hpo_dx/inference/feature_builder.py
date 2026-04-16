import pandas as pd


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