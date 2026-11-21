from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "student-por.csv"
REPORTS = ROOT / "reports"
PASS_MARK = 10
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]


def main() -> None:
    data = pd.read_csv(DATA_FILE, sep=";")
    X = data[FEATURES]
    # 1 means Fail, so precision, recall, and F1 describe failure detection.
    y = (data["G3"] < PASS_MARK).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "Decision tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
    }

    tuned_forest = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 3],
        },
        scoring="f1",
        cv=5,
        n_jobs=1,
    )
    tuned_forest.fit(X_train, y_train)
    models["Tuned random forest"] = tuned_forest.best_estimator_

    rows = []
    predictions = {}
    for name, model in models.items():
        if name != "Tuned random forest":
            model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Fail precision": precision_score(
                    y_test, predicted, zero_division=0
                ),
                "Fail recall": recall_score(y_test, predicted, zero_division=0),
                "Fail F1": f1_score(y_test, predicted, zero_division=0),
            }
        )

    results = pd.DataFrame(rows).sort_values("Fail F1", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "model_metrics.csv", index=False)

    ax = results.set_index("Model")[["Accuracy", "Fail F1"]].plot(
        kind="bar", figsize=(10, 5), ylim=(0, 1)
    )
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(["Accuracy", "Fail F1"])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / "model_comparison.png", dpi=160)
    plt.close()

    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for axis, (name, predicted) in zip(axes.flat, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predicted,
            labels=[0, 1],
            display_labels=["Pass", "Fail"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(name)
    for axis in axes.flat[len(predictions) :]:
        axis.axis("off")
    figure.suptitle("Test-set confusion matrices")
    figure.tight_layout()
    figure.savefig(REPORTS / "confusion_matrices.png", dpi=160)
    plt.close(figure)

    print("Dataset: UCI Student Performance (Portuguese course)")
    print(f"Records: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Pass: G3 >= {PASS_MARK}; Fail: G3 < {PASS_MARK}")
    print("Fail is the positive class for precision, recall, and F1.")
    print(f"Best random forest settings: {tuned_forest.best_params_}")
    print("\nTest results (higher scores are better):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nReports saved in: {REPORTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
