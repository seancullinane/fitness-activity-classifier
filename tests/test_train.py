from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_har_accuracy():
    """Test that RandomForest achieves >=90% accuracy on HAR test set."""
    root = Path("data/raw") / "UCI HAR Dataset"

    # Load train/test
    X_train = pd.read_csv(root/"train"/"X_train.txt", delim_whitespace=True, header=None).values
    y_train = pd.read_csv(root/"train"/"y_train.txt", delim_whitespace=True, header=None).values.ravel()
    X_test  = pd.read_csv(root/"test"/"X_test.txt",  delim_whitespace=True, header=None).values
    y_test  = pd.read_csv(root/"test"/"y_test.txt",  delim_whitespace=True, header=None).values.ravel()

    # Train small RF (fast for testing)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # Assert accuracy
    assert acc >= 0.90, f"Expected >= 0.90, got {acc:.3f}"
