"""
Pipeline lengkap: EDA -> Preprocessing -> Modeling (Decision Tree & Random Forest)
-> Tuning -> Evaluation -> SHAP -> Save model
Semua angka dihasilkan dari eksekusi nyata pada dataset_phishing.csv (11430 baris, 89 kolom)
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix, roc_curve, roc_auc_score)

RESULTS = {}
IMG = "../imgs/"

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("../data/raw/dataset_phishing.csv")
RESULTS["n_rows"], RESULTS["n_cols"] = df.shape
RESULTS["missing_total"] = int(df.isnull().sum().sum())
RESULTS["duplicates"] = int(df.duplicated().sum())
RESULTS["class_counts"] = df["status"].value_counts().to_dict()

# ============================================================
# 2. EDA
# ============================================================
# 2.1 Distribusi kelas
plt.figure(figsize=(5, 4))
sns.countplot(x="status", data=df, palette=["#E63946", "#2E86AB"])
plt.title("Distribusi Kelas Target (status)")
plt.tight_layout()
plt.savefig(IMG + "class_distribution.png", dpi=150)
plt.close()

# 2.2 Duplikat & outlier check (IQR) pada beberapa fitur numerik kunci
num_cols_check = ["length_url", "length_hostname", "nb_hyperlinks", "web_traffic", "domain_age"]
outlier_summary = {}
for c in num_cols_check:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_summary[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
RESULTS["outlier_summary"] = outlier_summary

# 2.3 Distribusi length_url per kelas (univariat)
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="length_url", hue="status", bins=40, kde=True, palette=["#E63946", "#2E86AB"])
plt.title("Distribusi Panjang URL berdasarkan Status")
plt.tight_layout()
plt.savefig(IMG + "length_url_distribution.png", dpi=150)
plt.close()
RESULTS["mean_length_url_phishing"] = float(df[df.status == "phishing"]["length_url"].mean())
RESULTS["mean_length_url_legitimate"] = float(df[df.status == "legitimate"]["length_url"].mean())

# 2.4 Correlation heatmap (multivariat) - subset fitur paling relevan
key_feats = ["length_url", "nb_hyperlinks", "web_traffic", "domain_age", "google_index",
             "page_rank", "nb_www", "phish_hints", "ratio_extHyperlinks", "safe_anchor"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[key_feats].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix - Fitur Kunci")
plt.tight_layout()
plt.savefig(IMG + "correlation_matrix.png", dpi=150)
plt.close()

# 2.5 google_index vs status (categorical insight)
plt.figure(figsize=(5.5, 4))
ctab = pd.crosstab(df["google_index"], df["status"], normalize="index") * 100
ctab.plot(kind="bar", stacked=True, color=["#E63946", "#2E86AB"], ax=plt.gca())
plt.title("Proporsi Status berdasarkan google_index")
plt.ylabel("Persentase (%)")
plt.xlabel("google_index (0=tidak terindeks, 1=terindeks)")
plt.tight_layout()
plt.savefig(IMG + "google_index_vs_status.png", dpi=150)
plt.close()

# ============================================================
# 3. PREPROCESSING
# ============================================================
le = LabelEncoder()
df["status_encoded"] = le.fit_transform(df["status"])  # legitimate=0, phishing=1
RESULTS["label_mapping"] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

df_model = df.drop(columns=["url", "status"])
X = df_model.drop(columns=["status_encoded"])
y = df_model["status_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
RESULTS["train_size"] = len(X_train)
RESULTS["test_size"] = len(X_test)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
joblib.dump(scaler, "../models/scaler.pkl")

# ============================================================
# 4. MODELING - DECISION TREE
# ============================================================
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
y_pred_dt = dt_base.predict(X_test)

param_grid_dt = {
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring="accuracy", n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
y_pred_best_dt = best_dt.predict(X_test)
y_prob_best_dt = best_dt.predict_proba(X_test)[:, 1]

# ============================================================
# 5. MODELING - RANDOM FOREST
# ============================================================
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)

param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
y_prob_best_rf = best_rf.predict_proba(X_test)[:, 1]

joblib.dump(best_rf, "../models/best_random_forest.pkl")
joblib.dump(best_dt, "../models/best_decision_tree.pkl")
joblib.dump(list(X.columns), "../models/feature_columns.pkl")

# ============================================================
# 6. EVALUATION SUMMARY
# ============================================================
def metrics_dict(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

RESULTS["metrics"] = {
    "Decision Tree (Baseline)": metrics_dict(y_test, y_pred_dt, dt_base.predict_proba(X_test)[:, 1]),
    "Decision Tree (Tuned)": metrics_dict(y_test, y_pred_best_dt, y_prob_best_dt),
    "Random Forest (Baseline)": metrics_dict(y_test, y_pred_rf, rf_base.predict_proba(X_test)[:, 1]),
    "Random Forest (Tuned)": metrics_dict(y_test, y_pred_best_rf, y_prob_best_rf),
}
RESULTS["best_params_dt"] = grid_dt.best_params_
RESULTS["best_params_rf"] = grid_rf.best_params_
RESULTS["cv_best_score_dt"] = float(grid_dt.best_score_)
RESULTS["cv_best_score_rf"] = float(grid_rf.best_score_)

cv_scores_rf = cross_val_score(best_rf, X, y, cv=10, scoring="accuracy")
RESULTS["cv10_mean_rf"] = float(cv_scores_rf.mean())
RESULTS["cv10_std_rf"] = float(cv_scores_rf.std())

# Classification reports (text)
RESULTS["classification_report_best_rf"] = classification_report(y_test, y_pred_best_rf, target_names=["legitimate", "phishing"])
RESULTS["classification_report_best_dt"] = classification_report(y_test, y_pred_best_dt, target_names=["legitimate", "phishing"])

# ============================================================
# 7. VISUALIZATIONS - CONFUSION MATRIX, ROC
# ============================================================
cm_rf = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(5, 4.2))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Prediksi"); plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest (Tuned)")
plt.tight_layout()
plt.savefig(IMG + "confusion_matrix_rf.png", dpi=150)
plt.close()
RESULTS["confusion_matrix_rf"] = cm_rf.tolist()

cm_dt = confusion_matrix(y_test, y_pred_best_dt)
plt.figure(figsize=(5, 4.2))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Prediksi"); plt.ylabel("Aktual")
plt.title("Confusion Matrix - Decision Tree (Tuned)")
plt.tight_layout()
plt.savefig(IMG + "confusion_matrix_dt.png", dpi=150)
plt.close()
RESULTS["confusion_matrix_dt"] = cm_dt.tolist()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_best_rf)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_best_dt)
plt.figure(figsize=(6.5, 5))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={RESULTS['metrics']['Random Forest (Tuned)']['auc']:.4f})", color="#2E86AB")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={RESULTS['metrics']['Decision Tree (Tuned)']['auc']:.4f})", color="#E63946")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Kurva ROC - Perbandingan Model")
plt.legend()
plt.tight_layout()
plt.savefig(IMG + "roc_comparison.png", dpi=150)
plt.close()

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
fi_rf = pd.DataFrame({"Feature": X.columns, "Importance": best_rf.feature_importances_}).sort_values("Importance", ascending=False)
plt.figure(figsize=(9, 7))
sns.barplot(data=fi_rf.head(15), x="Importance", y="Feature", color="#2E86AB")
plt.title("Top 15 Feature Importance - Random Forest (Tuned)")
plt.tight_layout()
plt.savefig(IMG + "feature_importance_rf.png", dpi=150)
plt.close()
RESULTS["top_features_rf"] = fi_rf.head(10).to_dict("records")

# ============================================================
# 9. SHAP (interpretability)
# ============================================================
import shap
explainer = shap.TreeExplainer(best_rf)
sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
if sv.ndim == 3:
    sv = sv[:, :, 1]

plt.figure()
shap.summary_plot(sv, sample, plot_type="bar", show=False, max_display=12)
plt.tight_layout()
plt.savefig(IMG + "shap_summary_bar.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(sv, sample, show=False, max_display=12)
plt.tight_layout()
plt.savefig(IMG + "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 10. MODEL COMPARISON CHART
# ============================================================
comp_df = pd.DataFrame(RESULTS["metrics"]).T.reset_index().rename(columns={"index": "Model"})
comp_df.to_csv("../data/processed/model_comparison.csv", index=False)

melt = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
sns.barplot(data=melt, x="Metric", y="Score", hue="Model")
plt.ylim(0.7, 1.0)
plt.title("Perbandingan Performa Semua Model")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(IMG + "model_comparison.png", dpi=150)
plt.close()

# ============================================================
# SAVE RESULTS JSON
# ============================================================
with open("../data/processed/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("PIPELINE SELESAI")
print(json.dumps({k: v for k, v in RESULTS.items() if k not in
                   ["classification_report_best_rf", "classification_report_best_dt", "top_features_rf"]}, indent=2, default=str))
