import json
import xgboost as xgb
from pathlib import Path




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
