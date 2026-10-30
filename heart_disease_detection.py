"""Educational heart-disease classification demo using 5-fold CV.

This project is for learning purposes only and must not be used for diagnosis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Heart_Disease_Detection.csv"
TARGET = "Heart_Disease"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    print("Dataset preview:\n", df.head())
    print("\nDataset information:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False)
    y = df[TARGET]
    if set(pd.Series(y.dropna().unique()).tolist()) != {0, 1}:
        raise ValueError("This script expects a binary Heart_Disease target encoded as 0 and 1.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, error_score="raise")

    print("\nFold-wise scores:")
    for i in range(5):
        print(f"Fold {i + 1}: Accuracy={scores['test_accuracy'][i]:.3f}, "
              f"Precision={scores['test_precision'][i]:.3f}, "
              f"Recall={scores['test_recall'][i]:.3f}, F1={scores['test_f1'][i]:.3f}")

    print("\nMean +/- standard deviation:")
    for metric in scoring:
        values = scores[f"test_{metric}"]
        print(f"{metric.capitalize()}: {values.mean():.3f} +/- {values.std():.3f}")

    oof_pred = cross_val_predict(pipeline, X, y, cv=cv)
    print("\nOut-of-fold classification report:\n",
          classification_report(y, oof_pred, zero_division=0))
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    print("Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):\n", cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease (0)", "Disease (1)"],
                yticklabels=["No Disease (0)", "Disease (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Heart Disease - Out-of-Fold Confusion Matrix")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "heart_disease_oof_confusion.png", dpi=150)
    plt.close()

    print("\nRecall for class 1 measures how many positive-class cases were detected.")
    print("False negatives are positive-class cases predicted as class 0.")
    print("Educational demonstration only - not for clinical diagnosis.")


if __name__ == "__main__":
    main()
