"""
Steel Plate Fault Lens - offline training script.

Trains five classifiers on the UCI "Steel Plates Faults" dataset, evaluates
them on a stratified hold-out split and stores everything the Streamlit app
needs inside ``model/``. The app never retrains: it only loads these files.

Run:  python train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV = PROJECT_ROOT / "data" / "steel_plates_faults_raw.csv"
ARTIFACT_DIR = PROJECT_ROOT / "model"
HOLDOUT_CSV = PROJECT_ROOT / "test_data.csv"

SEED = 2026
HOLDOUT_FRACTION = 0.20
TARGET_COLUMN = "fault_type"

# The raw file one-hot encodes the 7 fault classes in its last 7 columns.
FAULT_FLAG_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]


def load_plate_records(csv_path=RAW_CSV):
    """Read the raw UCI table and collapse the one-hot fault flags."""
    plates = pd.read_csv(csv_path)
    flag_sum = plates[FAULT_FLAG_COLUMNS].sum(axis=1)
    if not (flag_sum == 1).all():
        raise ValueError("Every plate must carry exactly one fault label.")
    plates[TARGET_COLUMN] = plates[FAULT_FLAG_COLUMNS].idxmax(axis=1)
    plates = plates.drop(columns=FAULT_FLAG_COLUMNS)
    return plates


def build_candidate_grid():
    """Each entry: (display name, file stem, pipeline, small search grid)."""
    return [
        (
            "Logistic Regression",
            "logistic_regression",
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]),
            {"clf__C": [0.1, 1.0, 10.0, 50.0]},
        ),
        (
            "Decision Tree",
            "decision_tree",
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 8, 12, 16],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        (
            "K-Nearest Neighbors",
            "knn",
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        (
            "Gaussian Naive Bayes",
            "naive_bayes",
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2]},
        ),
        (
            "Support Vector Machine",
            "svm",
            Pipeline([
                ("scale", StandardScaler()),
                # RBF SVC + Platt (sigmoid) calibration gives the class
                # probabilities needed for AUC (replaces probability=True,
                # which is deprecated from scikit-learn 1.9).
                ("clf", CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=SEED),
                    method="sigmoid", ensemble=False)),
            ]),
            {"clf__estimator__C": [1.0, 5.0, 10.0, 30.0],
             "clf__estimator__gamma": ["scale", 0.01, 0.05]},
        ),
    ]


def score_predictions(y_true, y_pred, y_proba, class_ids):
    """Six assignment metrics. Multi-class => macro averages, OvR AUC."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr",
                             average="macro", labels=class_ids),
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    plates = load_plate_records()
    feature_names = [c for c in plates.columns if c != TARGET_COLUMN]
    print(f"Loaded {plates.shape[0]} plates x {len(feature_names)} features")
    print(plates[TARGET_COLUMN].value_counts().to_string(), "\n")

    fault_encoder = LabelEncoder().fit(plates[TARGET_COLUMN])
    X = plates[feature_names]
    y = fault_encoder.transform(plates[TARGET_COLUMN])
    class_ids = np.arange(len(fault_encoder.classes_))

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, stratify=y, random_state=SEED
    )

    # Hold-out rows are what users upload to the app (features + true label).
    holdout_frame = X_hold.copy()
    holdout_frame[TARGET_COLUMN] = fault_encoder.inverse_transform(y_hold)
    holdout_frame.to_csv(HOLDOUT_CSV, index=False)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    scoreboard = {}
    chosen_params = {}
    for display_name, stem, pipeline, grid in build_candidate_grid():
        search = GridSearchCV(pipeline, grid, cv=folds, scoring="f1_macro",
                              n_jobs=-1)
        search.fit(X_fit, y_fit)
        tuned = search.best_estimator_

        hold_pred = tuned.predict(X_hold)
        hold_proba = tuned.predict_proba(X_hold)
        scoreboard[display_name] = score_predictions(
            y_hold, hold_pred, hold_proba, class_ids
        )
        chosen_params[display_name] = {
            k.split("__")[-1]: v for k, v in search.best_params_.items()
        }
        joblib.dump(tuned, ARTIFACT_DIR / f"{stem}.joblib")
        print(f"{display_name:<24} cv-F1={search.best_score_:.4f} "
              f"params={chosen_params[display_name]}")

    joblib.dump(fault_encoder, ARTIFACT_DIR / "fault_label_encoder.joblib")

    manifest = {
        "dataset": "UCI Steel Plates Faults (id 198)",
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_names,
        "fault_classes": fault_encoder.classes_.tolist(),
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "train_rows": int(len(X_fit)),
        "holdout_rows": int(len(X_hold)),
        "model_files": {
            name: f"{stem}.joblib"
            for name, stem, _, _ in build_candidate_grid()
        },
        "best_params": chosen_params,
        "holdout_metrics": {
            name: {m: round(float(v), 4) for m, v in vals.items()}
            for name, vals in scoreboard.items()
        },
    }
    with open(ARTIFACT_DIR / "training_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    table = pd.DataFrame(scoreboard).T.round(4)
    print("\nHold-out metrics (macro averaged, OvR AUC):")
    print(table.to_string())


if __name__ == "__main__":
    main()
