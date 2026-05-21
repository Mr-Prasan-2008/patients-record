# ============================================================
#  Real-World Data Project — Health Domain (Patient Records)
#  Dataset: Heart Disease Prediction (UCI Heart Disease Dataset)
#  Tools: pandas, numpy, matplotlib, seaborn, sklearn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ── 0. Styling ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e293b",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})
ACCENT   = "#38bdf8"
DANGER   = "#f43f5e"
SUCCESS  = "#34d399"
WARN     = "#fbbf24"
PURPLE   = "#a78bfa"

# ── 1. Simulate realistic Patient Dataset ─────────────────────
np.random.seed(42)
n = 303          # same size as original UCI dataset

age       = np.random.randint(29, 78, n)
sex       = np.random.randint(0, 2, n)         # 0=F, 1=M
cp        = np.random.randint(0, 4, n)         # chest pain type
trestbps  = np.random.randint(90, 200, n)      # resting BP
chol      = np.random.randint(126, 565, n)     # cholesterol
fbs       = (np.random.rand(n) > 0.85).astype(int)
restecg   = np.random.randint(0, 3, n)
thalach   = np.random.randint(71, 203, n)      # max heart rate
exang     = np.random.randint(0, 2, n)
oldpeak   = np.round(np.random.uniform(0, 6.2, n), 1)
slope     = np.random.randint(0, 3, n)
ca        = np.random.randint(0, 4, n)
thal      = np.random.randint(0, 4, n)

# Target: realistic correlation with risk factors
risk_score = (
    0.05 * (age - 50) +
    0.4  * sex +
    0.3  * (cp == 0).astype(int) +
    0.02 * (trestbps - 130) +
    0.01 * (chol - 250) +
    0.5  * exang +
    0.4  * oldpeak +
    -0.01* (thalach - 150) +
    np.random.normal(0, 0.5, n)
)
target = (risk_score > 0.5).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp,
    "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "thal": thal, "target": target
})

print("=" * 55)
print("   HEART DISEASE PREDICTION — DATA SCIENCE PROJECT")
print("=" * 55)
print(f"\n📋 Dataset Shape : {df.shape}")
print(f"🎯 Disease Cases : {df['target'].sum()} / {len(df)}"
      f"  ({df['target'].mean()*100:.1f}%)")
print(f"\n📊 Basic Statistics:\n{df.describe().round(2)}")
print(f"\n🔍 Missing Values: {df.isnull().sum().sum()} (None)")

# ── 2. EDA Figure ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("❤️  Patient Records — Exploratory Data Analysis",
             fontsize=18, fontweight="bold", color="#e2e8f0", y=0.98)
fig.patch.set_facecolor("#0f172a")

colors_bi = [SUCCESS, DANGER]
labels_bi = ["No Disease", "Heart Disease"]

# 2a. Age Distribution by Target
ax = axes[0, 0]
for t, c, lbl in zip([0, 1], colors_bi, labels_bi):
    ax.hist(df[df.target == t]["age"], bins=18, alpha=0.75,
            color=c, label=lbl, edgecolor="#0f172a")
ax.set_title("Age Distribution", color=ACCENT, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.legend(facecolor="#1e293b", edgecolor="#334155")

# 2b. Gender vs Disease
ax = axes[0, 1]
gender_disease = df.groupby(["sex", "target"]).size().unstack()
gender_disease.plot(kind="bar", ax=ax, color=colors_bi,
                    edgecolor="#0f172a", width=0.6)
ax.set_title("Gender vs Heart Disease", color=ACCENT, fontweight="bold")
ax.set_xticklabels(["Female", "Male"], rotation=0)
ax.set_xlabel(""); ax.set_ylabel("Count")
ax.legend(labels_bi, facecolor="#1e293b", edgecolor="#334155")

# 2c. Cholesterol Box Plot
ax = axes[0, 2]
data_chol = [df[df.target == 0]["chol"], df[df.target == 1]["chol"]]
bp = ax.boxplot(data_chol, patch_artist=True,
                medianprops=dict(color="#0f172a", linewidth=2))
for patch, c in zip(bp["boxes"], colors_bi):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_title("Cholesterol Levels", color=ACCENT, fontweight="bold")
ax.set_xticklabels(labels_bi); ax.set_ylabel("mg/dL")

# 2d. Max Heart Rate vs Age (scatter)
ax = axes[1, 0]
sc = ax.scatter(df["age"], df["thalach"],
                c=df["target"].map({0: SUCCESS, 1: DANGER}),
                alpha=0.6, edgecolors="none", s=40)
ax.set_title("Max Heart Rate vs Age", color=ACCENT, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate")
p0 = mpatches.Patch(color=SUCCESS, label="No Disease")
p1 = mpatches.Patch(color=DANGER,  label="Heart Disease")
ax.legend(handles=[p0, p1], facecolor="#1e293b", edgecolor="#334155")

# 2e. Chest Pain Type
ax = axes[1, 1]
cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_counts.plot(kind="bar", ax=ax, color=colors_bi,
               edgecolor="#0f172a", width=0.6)
ax.set_title("Chest Pain Type vs Disease", color=ACCENT, fontweight="bold")
ax.set_xticklabels([f"Type {i}" for i in range(4)], rotation=0)
ax.set_xlabel(""); ax.set_ylabel("Count")
ax.legend(labels_bi, facecolor="#1e293b", edgecolor="#334155")

# 2f. Correlation Heatmap
ax = axes[1, 2]
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm",
            annot=False, linewidths=0.3,
            cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation", color=ACCENT, fontweight="bold")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/01_eda.png",
            dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("\n✅ EDA plot saved.")

# ── 3. Preprocessing & Model Training ─────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_pred  = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_acc   = accuracy_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_pred)

print(f"\n🤖 Logistic Regression Accuracy : {lr_acc*100:.2f}%")
print(f"🌲 Random Forest Accuracy        : {rf_acc*100:.2f}%")
print(f"\n📋 Random Forest Classification Report:\n")
print(classification_report(y_test, rf_pred,
      target_names=["No Disease", "Heart Disease"]))

# ── 4. Results Figure ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("🤖  Model Evaluation — Heart Disease Prediction",
             fontsize=16, fontweight="bold", color="#e2e8f0")
fig.patch.set_facecolor("#0f172a")

# 4a. Confusion Matrix — RF
ax = axes[0]
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=ax,
            cmap="Blues", linewidths=2,
            xticklabels=labels_bi, yticklabels=labels_bi,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Confusion Matrix\n(Random Forest)",
             color=ACCENT, fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# 4b. ROC Curves
ax = axes[1]
for proba, name, color in [
    (lr_proba, f"Logistic Reg (AUC={auc(*roc_curve(y_test, lr_proba)[:2]):.2f})", PURPLE),
    (rf_proba, f"Random Forest (AUC={auc(*roc_curve(y_test, rf_proba)[:2]):.2f})", ACCENT),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=name)
ax.plot([0, 1], [0, 1], ":", color="#475569", linewidth=1.5)
ax.set_title("ROC Curves", color=ACCENT, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=9)
ax.fill_between([0,1],[0,1], alpha=0.05, color="#475569")

# 4c. Feature Importance — RF
ax = axes[2]
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", ax=ax,
    color=[ACCENT if v > importance.median() else "#334155"
           for v in importance.sort_values()],
    edgecolor="#0f172a")
ax.set_title("Feature Importance\n(Random Forest)",
             color=ACCENT, fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/02_model_results.png",
            dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ Model results plot saved.")

# ── 5. Save clean dataset ─────────────────────────────────────
df.to_csv("/mnt/user-data/outputs/heart_disease_dataset.csv", index=False)
print("✅ Dataset CSV saved.")
print("\n🎉 Project Complete! All outputs ready.")