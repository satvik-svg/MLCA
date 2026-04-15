from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"
MODEL_PATH = ROOT / "model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
METRICS_PATH = ROOT / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT / "feature_importance.csv"


def generate_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    interests = np.array(["STEM", "Arts", "Language", "Commerce"])
    chosen_interest = rng.choice(interests, size=n_samples, p=[0.35, 0.2, 0.25, 0.2])

    math = np.clip(rng.normal(68, 15, n_samples), 20, 100)
    science = np.clip(rng.normal(66, 14, n_samples), 20, 100)
    english = np.clip(rng.normal(70, 13, n_samples), 20, 100)
    study_time = np.clip(rng.normal(3.5, 1.3, n_samples), 0.5, 9.5)

    # Interest nudges certain subjects and creates a realistic signal for the model.
    math = np.where(chosen_interest == "STEM", math + rng.normal(5, 2, n_samples), math)
    science = np.where(chosen_interest == "STEM", science + rng.normal(4, 2, n_samples), science)
    english = np.where(chosen_interest == "Language", english + rng.normal(6, 2, n_samples), english)

    english = np.where(chosen_interest == "Arts", english + rng.normal(3, 1.5, n_samples), english)
    science = np.where(chosen_interest == "Commerce", science - rng.normal(2, 1, n_samples), science)

    # Lower study time usually affects all subjects; Math often drops slightly more.
    fatigue = np.clip(3.0 - study_time, 0, None)
    math = np.clip(math - (fatigue * rng.normal(3.0, 0.8, n_samples)), 0, 100)
    science = np.clip(science - (fatigue * rng.normal(2.4, 0.7, n_samples)), 0, 100)
    english = np.clip(english - (fatigue * rng.normal(2.0, 0.6, n_samples)), 0, 100)

    marks = np.vstack([math, science, english]).T
    weak_idx = np.argmin(marks, axis=1)
    labels = np.array(["Mathematics", "Science", "English"])[weak_idx]

    df = pd.DataFrame(
        {
            "math_marks": np.round(math, 1),
            "science_marks": np.round(science, 1),
            "english_marks": np.round(english, 1),
            "study_time_hours": np.round(study_time, 1),
            "interest": chosen_interest,
            "weak_subject": labels,
        }
    )

    # Inject sparse missing values to test imputation in preprocessing.
    for col in ["math_marks", "science_marks", "english_marks", "study_time_hours", "interest"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["math_marks", "science_marks", "english_marks", "study_time_hours"]
    categorical_features = ["interest"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = pipeline.predict(x_test)
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    return result


def train_and_save() -> None:
    if not DATA_PATH.exists():
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    x = df[["math_marks", "science_marks", "english_marks", "study_time_hours", "interest"]]
    y = df["weak_subject"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    log_reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    log_reg_pipeline.fit(x_train, y_train)
    lr_metrics = evaluate_pipeline("LogisticRegression", log_reg_pipeline, x_test, y_test)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [120, 200],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1_weighted",
        verbose=0,
    )
    grid.fit(x_train, y_train)

    best_rf_pipeline: Pipeline = grid.best_estimator_
    rf_metrics = evaluate_pipeline("RandomForest", best_rf_pipeline, x_test, y_test)

    # As requested in the guide, Random Forest is the final selected model.
    joblib.dump(best_rf_pipeline, MODEL_PATH)

    # Save a standalone scaler artifact for numeric features.
    scaler_only = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaler_only.fit(x_train[["math_marks", "science_marks", "english_marks", "study_time_hours"]])
    joblib.dump(scaler_only, SCALER_PATH)

    preprocessor_fitted: ColumnTransformer = best_rf_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    rf_model: RandomForestClassifier = best_rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics_payload = {
        "final_model": "RandomForest",
        "best_params": grid.best_params_,
        "logistic_regression": {
            "accuracy": lr_metrics["accuracy"],
            "f1_weighted": lr_metrics["f1_weighted"],
        },
        "random_forest": {
            "accuracy": rf_metrics["accuracy"],
            "f1_weighted": rf_metrics["f1_weighted"],
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Training completed.")
    print(f"Logistic Regression -> Accuracy: {lr_metrics['accuracy']:.4f}, F1: {lr_metrics['f1_weighted']:.4f}")
    print(f"Random Forest      -> Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1_weighted']:.4f}")
    print(f"Best RF params: {grid.best_params_}")
    print(f"Saved: {MODEL_PATH.name}, {SCALER_PATH.name}, {METRICS_PATH.name}, {FEATURE_IMPORTANCE_PATH.name}")


if __name__ == "__main__":
    train_and_save()
