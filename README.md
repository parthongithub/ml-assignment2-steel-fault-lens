# Steel Plate Fault Lens 🔩

Multi-class classification of surface faults on steel plates using five
classical machine-learning models, served through an interactive Streamlit
web application.

- **GitHub repository:** <https://github.com/parthongithub/ml-assignment2-steel-fault-lens>
- **Live Streamlit app:** <https://2026aj05442-ml-assignment2.streamlit.app>

---

## a. Problem statement

In a steel rolling mill every plate is photographed and the detected defect
region is described by a set of geometric and luminosity measurements. A
quality engineer must decide **which of seven fault types** the plate has so
that it can be routed to the right repair or scrap process. Doing this by hand
is slow and inconsistent.

The goal of this project is to **predict the fault type of a steel plate from
its 27 numeric inspection measurements**, compare five different classifiers
on the same data using six evaluation metrics, and deploy the trained models
in a web app where a user can upload new (test) records and immediately see
predictions, metrics, a confusion matrix and a classification report.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Steel Plates Faults |
| Source | UCI Machine Learning Repository, dataset id 198 (Semeion Research Center, Rome) – <https://archive.ics.uci.edu/dataset/198/steel+plates+faults> |
| Licence | CC BY 4.0 |
| Instances | 1,941 steel plates (no missing values, no duplicate rows) |
| Features | 27 numeric input features |
| Target | `fault_type` – 7 classes (multi-class) |
| Train / test split | Stratified 80 % / 20 % (1,552 / 389 rows), `random_state = 2026` |

**Features (27).** They fall into four groups:

- **Location and size of the defect:** `X_Minimum`, `X_Maximum`, `Y_Minimum`,
  `Y_Maximum`, `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `LogOfAreas`,
  `Log_X_Index`, `Log_Y_Index`, `SigmoidOfAreas`
- **Brightness of the defect:** `Sum_of_Luminosity`, `Minimum_of_Luminosity`,
  `Maximum_of_Luminosity`, `Luminosity_Index`
- **Plate / process information:** `Length_of_Conveyer`, `TypeOfSteel_A300`,
  `TypeOfSteel_A400`, `Steel_Plate_Thickness`
- **Shape indices:** `Edges_Index`, `Empty_Index`, `Square_Index`,
  `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`,
  `Orientation_Index`

**Target classes and their counts.** In the original file the target is
stored as 7 one-hot columns. These were merged into a single `fault_type`
label.

| Fault type | Count | Share |
|---|---:|---:|
| Other_Faults | 673 | 34.7 % |
| Bumps | 402 | 20.7 % |
| K_Scatch | 391 | 20.1 % |
| Z_Scratch | 190 | 9.8 % |
| Pastry | 158 | 8.1 % |
| Stains | 72 | 3.7 % |
| Dirtiness | 55 | 2.8 % |

The classes are clearly **imbalanced**: the largest class is about 12 times
the smallest. For this reason Precision, Recall and F1 are reported as
**macro averages**, where every fault type counts equally. AUC is the **macro
one-vs-rest** ROC-AUC.

## c. Models used

All five models are trained on the same training split.
- **Feature scaling:** a `StandardScaler` is used for every model except the
  Decision Tree, which does not need scaling. The scaler is stored *inside*
  each saved pipeline, so the app applies exactly the same preprocessing.
- **Tuning:** hyper-parameters were chosen with 5-fold stratified
  `GridSearchCV`, using macro-F1 as the selection score.

| # | Model | Tuned hyper-parameters |
|---|---|---|
| 1 | Logistic Regression | `C = 50` (L2 penalty, lbfgs) |
| 2 | Decision Tree Classifier | `criterion = entropy`, `max_depth = 12`, `min_samples_leaf = 3` |
| 3 | K-Nearest Neighbors | `k = 7`, `weights = distance` |
| 4 | Naive Bayes (Gaussian) | `var_smoothing = 0.01` |
| 5 | Support Vector Machine | RBF kernel, `C = 30`, `gamma = 0.01`, with Platt (sigmoid) calibration for class probabilities |

### Comparison table (hold-out test set, 389 plates)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6992 | 0.9340 | 0.7193 | 0.7190 | 0.7156 | 0.6124 |
| Decision Tree | 0.7404 | 0.8940 | 0.7682 | 0.7777 | 0.7719 | 0.6673 |
| kNN | 0.7378 | **0.9376** | 0.7894 | **0.7866** | 0.7858 | 0.6645 |
| Naive Bayes | 0.6144 | 0.9251 | 0.6244 | 0.7362 | 0.6496 | 0.5452 |
| SVM | **0.7481** | 0.9366 | **0.8060** | 0.7864 | **0.7956** | **0.6750** |

*Precision, Recall and F1 are macro-averaged. AUC is macro one-vs-rest.
The best value in each column is in bold.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A solid linear baseline. Its AUC (0.934) is high, so it ranks the classes well, but its accuracy (0.699) and MCC (0.612) are the second lowest. A single linear boundary per class cannot separate the overlapping geometric shapes of **Bumps** (F1 0.55) and **Other_Faults** (F1 0.64). Train accuracy (0.749) is close to test accuracy, so the model **underfits** rather than overfits. |
| Decision Tree | Second-best MCC (0.667) and the best F1 on **Z_Scratch** (0.92) and **Other_Faults** (0.68): simple threshold rules on width, area and orientation work well for these faults. However, its AUC (0.894) is the **lowest** of all models, because a tree outputs coarse leaf probabilities. The gap between train (0.917) and test (0.740) accuracy shows some overfitting, even with depth 12 and at least 3 samples per leaf. |
| kNN | After scaling, distance-weighted k = 7 gives the **best AUC (0.938)** and best macro recall (0.787). It handles the small classes very well (Dirtiness F1 0.82, Stains 1.00). It memorises the training set (train accuracy 1.00), and its predictions depend heavily on feature scaling. The saved model must keep all 1,552 training rows, so it is the largest and slowest model at prediction time. |
| Naive Bayes | The **weakest** model (accuracy 0.614, MCC 0.545, F1 0.650). Gaussian NB assumes that features are independent and normally distributed. Here 7 feature pairs have correlation |r| > 0.9 (for example X_Min/X_Max, area/perimeter/luminosity sum, and the two TypeOfSteel flags), and several features are heavily skewed (skewness of Y_Perimeter is about 39). Both assumptions are therefore broken. Its recall (0.736) is much higher than its precision (0.624): it over-predicts the minority faults (61 Pastry predictions vs 32 true, 24 Dirtiness vs 11) and finds only 64 of the 135 **Other_Faults** plates (F1 0.42). Its AUC (0.925) is still reasonable, so the class *ranking* is better than the hard decisions. |
| SVM | The best or joint-best model on **4 of 6 metrics**: accuracy 0.748, precision 0.806, F1 0.796 and MCC 0.675. Its AUC (0.937) and recall (0.786) are only about 0.001 behind kNN. The RBF kernel learns non-linear boundaries, and the train/test gap is small (0.837 vs 0.748), so it **generalises better than kNN or the Decision Tree**. It also gives the best F1 on Bumps, K_Scatch and Pastry. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF).** It has the highest MCC, the most reliable single score for imbalanced multi-class data, and also the highest accuracy, precision and macro-F1. kNN is a very close second and is slightly ahead on AUC and recall. The hardest pair to separate for **every** model is **Bumps vs Other_Faults**, the main source of errors in all confusion matrices. Naive Bayes is clearly last because its independence assumption does not hold for these correlated geometric features. |

---

## Streamlit app features

The app (`app.py`) **only loads the saved models from `model/`**; it never
retrains.

1. **CSV upload** of test data from the sidebar. A *Download sample
   test_data.csv* button is also provided.
2. **Model selection dropdown** to choose any of the five classifiers.
3. **Evaluation metrics** shown as metric cards: Accuracy, AUC, Precision,
   Recall, F1, MCC.
4. **Confusion matrix** (heat-map) and **classification report** for the
   selected model.
5. **All five models** tab: the same metrics computed live for every model on
   the uploaded file, a highlighted best-per-metric table, and a bar chart
   whose metric can be switched.
6. A downloadable **predictions** table with a per-row confidence value, plus
   a preview of the uploaded data and its class distribution.

If the uploaded CSV has no `fault_type` column, the app still predicts but
skips the metrics. Missing or non-numeric feature columns are reported with
a clear error message.

## Repository structure

```
steel-fault-lens/
├── app.py                      # Streamlit web application
├── train_models.py             # training + evaluation script (source code)
├── ML_Assignment2_Steel_Fault_Lens.ipynb  # same pipeline as a notebook, with EDA + plots
├── requirements.txt            # pinned dependencies for Streamlit Cloud
├── test_data.csv               # 20 % hold-out split (389 rows, with labels)
├── README.md
├── data/
│   └── steel_plates_faults_raw.csv   # original UCI table (1,941 rows)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── fault_label_encoder.joblib    # fault name <-> integer id
│   └── training_manifest.json        # features, classes, params, metrics
└── .streamlit/
    └── config.toml             # custom colour theme
```

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_models.py        # optional: re-creates model/ and test_data.csv
# or open ML_Assignment2_Steel_Fault_Lens.ipynb in Jupyter -> Restart & Run All
streamlit run app.py
```

Python 3.12 or newer is required, because of the pinned numpy and scipy
versions.
