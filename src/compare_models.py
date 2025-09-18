# src/compare_models.py
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HAR_ROOT = Path("data/raw") / "UCI HAR Dataset"

def load_har(root: Path):
    # Load feature matrices
    X_train = pd.read_csv(root/"train"/"X_train.txt", sep=r"\s+", header=None).values
    y_train = pd.read_csv(root/"train"/"y_train.txt", sep=r"\s+", header=None).values.ravel()

    X_test  = pd.read_csv(root/"test"/"X_test.txt",  sep=r"\s+", header=None).values
    y_test  = pd.read_csv(root/"test"/"y_test.txt",  sep=r"\s+", header=None).values.ravel()

    # 👇 Normalise labels: dataset uses 1–6, but XGBoost requires 0–5
    y_train = y_train - y_train.min()
    y_test  = y_test  - y_test.min()

    return X_train, y_train, X_test, y_test


def main():
    if not HAR_ROOT.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/get_data.py")

    X_train, y_train, X_test, y_test = load_har(HAR_ROOT)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "SVM (RBF)":    SVC(C=3.0, gamma="scale", probability=False, random_state=42),
        "LogReg": LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto"),
        "GradBoost":    GradientBoostingClassifier(random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
            tree_method="hist", objective="multi:softmax"
        )

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append({"model": name, "accuracy": acc})

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path.resolve())
    print(df)

    # Bar chart
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="model", y="accuracy")
    plt.ylim(0.80, 1.00)
    plt.title("HAR Model Comparison (Accuracy)")
    plt.tight_layout()
    fig_path = out_dir / "model_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved:", fig_path.resolve())

if __name__ == "__main__":
    main()
