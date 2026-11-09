"""
Steel Plate Fault Lens - Streamlit front end.

Loads the five pre-trained classifiers from ``model/`` (built by
train_models.py) and evaluates them on a CSV uploaded by the user.
Nothing is retrained here.
"""

import json
from io import StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

FAULT_BLURBS = {
    "Pastry": "small patches / pastry-like surface irregularities",
    "Z_Scratch": "scratches running along the rolling direction",
    "K_Scatch": "scratches across the rolling direction",
    "Stains": "discoloured stain regions",
    "Dirtiness": "dirt deposits on the surface",
    "Bumps": "raised bumps on the plate",
    "Other_Faults": "any defect outside the six named types",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(
    page_title="Steel Plate Fault Lens",
    page_icon="🔩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .lens-banner {
        background: linear-gradient(100deg, #1F2A36 0%, #3B5266 60%, #C8553D 100%);
        padding: 1.3rem 1.6rem; border-radius: 14px; color: #F7F5F2;
        margin-bottom: 1rem;
    }
    .lens-banner h1 { color: #F7F5F2; margin: 0; font-size: 2.0rem; }
    .lens-banner p  { margin: 0.3rem 0 0 0; opacity: 0.9; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #D5DCE3;
        border-left: 5px solid #C8553D; border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    </style>
    <div class="lens-banner">
      <h1>🔩 Steel Plate Fault Lens</h1>
      <p>Upload inspection records of steel plates and see how five
      pre-trained classifiers diagnose the surface fault type.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained classifiers ...")
def load_artifacts():
    with open(ARTIFACT_DIR / "training_manifest.json") as handle:
        manifest = json.load(handle)
    classifiers = {
        name: joblib.load(ARTIFACT_DIR / file_name)
        for name, file_name in manifest["model_files"].items()
    }
    encoder = joblib.load(ARTIFACT_DIR / "fault_label_encoder.joblib")
    return manifest, classifiers, encoder


def evaluate(y_true, y_pred, y_proba):
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # Macro one-vs-rest AUC over the fault classes present in the file
    # (identical to sklearn's multi_class="ovr" when all classes appear,
    # but still works when an uploaded file holds only a few classes).
    per_class_auc = [
        roc_auc_score(y_true == cls, y_proba[:, cls])
        for cls in np.unique(y_true)
        if 0 < (y_true == cls).sum() < len(y_true)
    ]
    scores["AUC"] = float(np.mean(per_class_auc)) if per_class_auc \
        else float("nan")
    return {k: scores[k] for k in METRIC_ORDER}


def draw_confusion(y_true, y_pred, class_names):
    grid = confusion_matrix(y_true, y_pred,
                            labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(grid, cmap="OrRd")
    ax.set_xticks(range(len(class_names)), class_names, rotation=40,
                  ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted fault")
    ax.set_ylabel("Actual fault")
    threshold = grid.max() / 2 if grid.max() else 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, r, grid[r, c], ha="center", va="center",
                    color="white" if grid[r, c] > threshold else "#1F2A36",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


manifest, classifiers, fault_encoder = load_artifacts()
feature_columns = manifest["feature_columns"]
target_column = manifest["target_column"]
fault_names = manifest["fault_classes"]

with st.sidebar:
    st.header("⚙️ Controls")
    chosen_model = st.selectbox(
        "Choose a classifier",
        list(classifiers.keys()),
        index=list(classifiers.keys()).index("Support Vector Machine"),
    )
    uploaded = st.file_uploader(
        "Upload test data (CSV)", type=["csv"],
        help=f"Needs the {len(feature_columns)} feature columns. Include a "
             f"'{target_column}' column to get metrics.",
    )
    st.download_button(
        "⬇️ Download sample test_data.csv",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        help="The untouched 20% hold-out split - use it to try the app.",
    )
    st.caption(
        f"Models were trained offline on {manifest['train_rows']} plates "
        f"(UCI dataset id 198). This app only runs inference."
    )

if uploaded is None:
    st.info("👈 Upload a CSV of plate records from the sidebar to begin. "
            "No file handy? Download the sample hold-out file first.")
    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("What the app recognises")
        st.table(pd.DataFrame(
            {"Fault type": list(FAULT_BLURBS),
             "Meaning": list(FAULT_BLURBS.values())}
        ).set_index("Fault type"))
    with right:
        st.subheader("Hold-out benchmark (from training)")
        st.dataframe(
            pd.DataFrame(manifest["holdout_metrics"]).T[METRIC_ORDER]
            .style.format("{:.4f}").highlight_max(color="#F4C7B8"),
            width="stretch",
        )
    st.stop()

try:
    records = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
except Exception as exc:  # malformed file
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in feature_columns if c not in records.columns]
if missing:
    st.error("The uploaded file is missing required feature columns: "
             + ", ".join(missing))
    st.stop()

plate_features = records[feature_columns].apply(pd.to_numeric,
                                                errors="coerce")
if plate_features.isna().any().any():
    st.error("Some feature values are empty or non-numeric. "
             "Please clean the file and upload again.")
    st.stop()

has_truth = target_column in records.columns
y_true = None
if has_truth:
    unknown = sorted(set(records[target_column]) - set(fault_names))
    if unknown:
        st.error(f"Unknown fault labels in '{target_column}': {unknown}")
        st.stop()
    y_true = fault_encoder.transform(records[target_column])

st.success(f"Loaded **{len(records)}** plate records from "
           f"`{uploaded.name}`.")

tab_single, tab_compare, tab_data = st.tabs(
    [f"🔍 {chosen_model}", "📊 All five models", "🗂️ Uploaded data"]
)

active = classifiers[chosen_model]
pred_ids = active.predict(plate_features)
pred_proba = active.predict_proba(plate_features)
pred_labels = fault_encoder.inverse_transform(pred_ids)

with tab_single:
    if has_truth:
        scores = evaluate(y_true, pred_ids, pred_proba)
        st.subheader(f"Evaluation metrics - {chosen_model}")
        cols = st.columns(6)
        for col, metric in zip(cols, METRIC_ORDER):
            value = scores[metric]
            col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")
        st.caption("Precision, Recall and F1 are macro averages over the "
                   "7 fault classes; AUC is macro one-vs-rest.")

        graph_col, report_col = st.columns([1, 1])
        with graph_col:
            st.markdown("**Confusion matrix**")
            st.pyplot(draw_confusion(y_true, pred_ids, fault_names))
        with report_col:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, pred_ids, labels=np.arange(len(fault_names)),
                target_names=fault_names, output_dict=True, zero_division=0,
            )
            overall_acc = report.pop("accuracy")
            report_frame = pd.DataFrame(report).T
            report_frame.loc["accuracy"] = [np.nan, np.nan, overall_acc,
                                            len(y_true)]
            report_frame["support"] = report_frame["support"].astype(int)
            st.dataframe(report_frame.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}",
                 "f1-score": "{:.3f}"}, na_rep=""), width="stretch")
    else:
        st.warning(f"No '{target_column}' column found, so only predictions "
                   "are shown (metrics need the true labels).")

    st.markdown("**Predictions**")
    outcome = pd.DataFrame({
        "predicted_fault": pred_labels,
        "confidence": pred_proba.max(axis=1).round(4),
    })
    if has_truth:
        outcome.insert(0, "actual_fault", records[target_column].values)
        outcome["correct"] = outcome["actual_fault"] == outcome[
            "predicted_fault"]
    st.dataframe(outcome, width="stretch", height=260)
    st.download_button(
        "⬇️ Download predictions",
        outcome.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen_model.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

with tab_compare:
    if not has_truth:
        st.warning("Upload a file with the true labels to compare models.")
    else:
        st.subheader("All five models on the uploaded file")
        board = {}
        for name, clf in classifiers.items():
            board[name] = evaluate(y_true, clf.predict(plate_features),
                                   clf.predict_proba(plate_features))
        board_frame = pd.DataFrame(board).T[METRIC_ORDER]
        st.dataframe(
            board_frame.style.format("{:.4f}")
            .highlight_max(color="#F4C7B8"),
            width="stretch",
        )
        leader = board_frame["F1"].idxmax()
        st.markdown(f"🏆 Highest macro-F1 on this file: **{leader}** "
                    f"({board_frame.loc[leader, 'F1']:.4f})")
        metric_pick = st.radio("Metric to chart", METRIC_ORDER, index=4,
                               horizontal=True)
        st.bar_chart(board_frame[[metric_pick]], color="#C8553D",
                     horizontal=True)

with tab_data:
    st.subheader("Preview of the uploaded file")
    st.dataframe(records.head(50), width="stretch")
    if has_truth:
        st.markdown("**Fault distribution in this file**")
        st.bar_chart(records[target_column].value_counts(), color="#3B5266")
