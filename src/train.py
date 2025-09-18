# src/train.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

HAR_ROOT = Path("data/raw") / "UCI HAR Dataset"

def load_har(root: Path):
    X_train = pd.read_csv(root/"train"/"X_train.txt", sep=r"\s+", header=None).values
    y_train = pd.read_csv(root/"train"/"y_train.txt", sep=r"\s+", header=None).values.ravel()
    X_test  = pd.read_csv(root/"test"/"X_test.txt",  sep=r"\s+", header=None).values
    y_test  = pd.read_csv(root/"test"/"y_test.txt",  sep=r"\s+", header=None).values.ravel()
    labels  = pd.read_csv(root/"activity_labels.txt", sep=r"\s+", header=None, names=["id","name"])
    id2name = dict(zip(labels["id"], labels["name"]))  # {1:'WALKING', ...}
    return X_train, y_train, X_test, y_test, id2name

def main():
    # sanity: print which file is running
    print("Running:", Path(__file__).resolve())

    if not HAR_ROOT.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/get_data.py")

    # 1) Load
    X_train, y_train, X_test, y_test, id2name = load_har(HAR_ROOT)

    # 2) Train
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 3) Predict & metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(id2name))
    print(f"Accuracy: {acc:.4f}\n")

    # 4) Ensure outputs/
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Save confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[id2name[i] for i in sorted(id2name)],
        yticklabels=[id2name[i] for i in sorted(id2name)]
    )
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"HAR Confusion Matrix (Acc: {acc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved:", (out_dir / "confusion_matrix.png").resolve())

    # 6) Save classification report (text)
    report_txt = classification_report(
        y_test, y_pred,
        target_names=[id2name[i] for i in sorted(id2name)]
    )
    (out_dir / "classification_report.txt").write_text(report_txt)
    print("Saved:", (out_dir / "classification_report.txt").resolve())

    # 7) Save feature importance plot (top 15)
    importances = clf.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(7,5))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [f"f{int(i)}" for i in idx])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved:", (out_dir / "feature_importance.png").resolve())

    # 8) Save model
    dump(clf, out_dir / "model.joblib")
    print("Saved:", (out_dir / "model.joblib").resolve())

if __name__ == "__main__":
    main()


