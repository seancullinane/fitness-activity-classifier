from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt, seaborn as sns
from joblib import dump

HAR_ROOT = Path("data/raw") / "UCI HAR Dataset"

def load_har(root: Path):
    X_train = pd.read_csv(root/"train"/"X_train.txt", delim_whitespace=True, header=None).values
    y_train = pd.read_csv(root/"train"/"y_train.txt", delim_whitespace=True, header=None).values.ravel()
    X_test  = pd.read_csv(root/"test"/"X_test.txt",  delim_whitespace=True, header=None).values
    y_test  = pd.read_csv(root/"test"/"y_test.txt",  delim_whitespace=True, header=None).values.ravel()
    labs = pd.read_csv(root/"activity_labels.txt", delim_whitespace=True, header=None, names=["id","name"])
    id2name = dict(zip(labs["id"], labs["name"]))
    return X_train, y_train, X_test, y_test, id2name

def main():
    if not HAR_ROOT.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/get_data.py")

    X_train, y_train, X_test, y_test, id2name = load_har(HAR_ROOT)

    # 1. Train model
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 2. Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:\n",
          classification_report(y_test, y_pred,
                                target_names=[id2name[i] for i in sorted(id2name)]))

    # 3. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(id2name))

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix PNG
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[id2name[i] for i in sorted(id2name)],
                yticklabels=[id2name[i] for i in sorted(id2name)])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"HAR Confusion Matrix (Acc: {acc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir/"confusion_matrix.png", dpi=150)
    plt.close()

    # Save trained model
    dump(clf, out_dir/"model.joblib")
    print("Saved:", (out_dir/"confusion_matrix.png").resolve())
    print("Saved:", (out_dir/"model.joblib").resolve())

if __name__ == "__main__":
    main()
