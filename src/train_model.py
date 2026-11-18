"""Train time-split models for 1/3/5-day news reactions."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_DIR, HORIZONS, MODEL_DIR, RANDOM_STATE, RESULTS_DIR


TEXT_COLUMN = "title"
CATEGORICAL_COLUMNS = ["ticker", "sector", "source_name", "event_type_rule"]
NUMERIC_COLUMNS = [
    "sentiment_score_rule",
    "publication_hour_msk",
    "publication_weekday",
    "publication_is_weekend",
    "publication_month",
]
FEATURE_COLUMNS = [TEXT_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=6000,
                    sublinear_tf=True,
                ),
                TEXT_COLUMN,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", numeric, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def make_regressor(alpha: float = 12.0) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor()),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def make_classifier(c_value: float = 0.7) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work[TEXT_COLUMN] = work[TEXT_COLUMN].fillna("").astype(str)
    for column in CATEGORICAL_COLUMNS:
        work[column] = work[column].fillna("unknown").astype(str)
    for column in NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["published_at_msk"] = pd.to_datetime(work["published_at_msk"], utc=True)
    return work.sort_values("published_at_msk")


def _time_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dates = sorted(frame["published_at_msk"].dt.date.unique())
    if len(dates) < 10:
        raise RuntimeError("Too few unique event dates for temporal validation")
    cutoff = dates[max(1, int(len(dates) * (1 - test_fraction)))]
    train = frame.loc[frame["published_at_msk"].dt.date < cutoff].copy()
    test = frame.loc[frame["published_at_msk"].dt.date >= cutoff].copy()
    if min(len(train), len(test)) < 30:
        raise RuntimeError(f"Temporal split is too small: train={len(train)}, test={len(test)}")
    return train, test, str(cutoff)


def _sector_metrics(test: pd.DataFrame, actual: np.ndarray, pred: np.ndarray, direction_pred: np.ndarray) -> dict:
    result = {}
    for sector, indices in test.groupby("sector").groups.items():
        loc = test.index.get_indexer(indices)
        y = actual[loc]
        p = pred[loc]
        d = direction_pred[loc]
        result[sector] = {
            "n": int(len(loc)),
            "mae": float(mean_absolute_error(y, p)),
            "direction_accuracy": float(accuracy_score((y > 0).astype(int), d)),
        }
    return result


def _select_parameters(train: pd.DataFrame, target: str) -> tuple[float, float, dict]:
    """Tune only inside the training period; the outer test remains untouched."""
    inner_train, inner_valid, inner_cutoff = _time_split(train, test_fraction=0.20)
    y_inner = inner_train[target].astype(float).to_numpy()
    y_valid = inner_valid[target].astype(float).to_numpy()

    reg_scores = {}
    for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0):
        model = make_regressor(alpha)
        model.fit(inner_train[FEATURE_COLUMNS], y_inner)
        reg_scores[alpha] = float(mean_absolute_error(y_valid, model.predict(inner_valid[FEATURE_COLUMNS])))
    best_alpha = min(reg_scores, key=lambda value: (reg_scores[value], -value))

    clf_scores = {}
    for c_value in (0.02, 0.05, 0.2, 0.7, 2.0):
        model = make_classifier(c_value)
        model.fit(inner_train[FEATURE_COLUMNS], (y_inner > 0).astype(int))
        clf_scores[c_value] = float(
            accuracy_score((y_valid > 0).astype(int), model.predict(inner_valid[FEATURE_COLUMNS]))
        )
    best_c = max(clf_scores, key=lambda value: (clf_scores[value], -value))
    details = {
        "inner_cutoff": inner_cutoff,
        "ridge_validation_mae": {str(k): v for k, v in reg_scores.items()},
        "logistic_validation_accuracy": {str(k): v for k, v in clf_scores.items()},
    }
    return float(best_alpha), float(best_c), details


def train_models() -> dict:
    frame = _prepare(pd.read_csv(DATA_DIR / "modeling_dataset.csv"))
    train, test, cutoff = _time_split(frame)
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    bundle = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_columns": FEATURE_COLUMNS,
            "target": "abnormal return relative to IMOEX",
            "evaluation_cutoff": cutoff,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "versions": versions,
        },
        "horizons": {},
    }
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "chronological",
        "cutoff_date": cutoff,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "versions": versions,
        "horizons": {},
        "interpretation": "Returns are decimal abnormal returns; multiply by 100 for percent.",
    }
    test_predictions = test[["event_id", "published_at_msk", "ticker", "sector", "title"]].copy()

    for horizon in HORIZONS:
        target = f"abnormal_return_{horizon}d"
        y_train = train[target].astype(float).to_numpy()
        y_test = test[target].astype(float).to_numpy()

        best_alpha, best_c, tuning = _select_parameters(train, target)

        reg_eval = make_regressor(best_alpha)
        clf_eval = make_classifier(best_c)
        reg_eval.fit(train[FEATURE_COLUMNS], y_train)
        clf_eval.fit(train[FEATURE_COLUMNS], (y_train > 0).astype(int))
        pred = reg_eval.predict(test[FEATURE_COLUMNS])
        direction_pred = clf_eval.predict(test[FEATURE_COLUMNS])
        probability = clf_eval.predict_proba(test[FEATURE_COLUMNS])[:, 1]

        zero_mae = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
        majority_class = int(np.mean(y_train > 0) >= 0.5)
        majority_accuracy = float(accuracy_score((y_test > 0).astype(int), np.full(len(y_test), majority_class)))
        residuals = y_test - pred
        residual_q10, residual_q90 = np.quantile(residuals, [0.10, 0.90])
        horizon_metrics = {
            "n_test": int(len(test)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
            "r2": float(r2_score(y_test, pred)),
            "direction_accuracy": float(accuracy_score((y_test > 0).astype(int), direction_pred)),
            "baseline_zero_mae": zero_mae,
            "baseline_majority_direction_accuracy": majority_accuracy,
            "mean_actual": float(np.mean(y_test)),
            "mean_predicted": float(np.mean(pred)),
            "residual_q10": float(residual_q10),
            "residual_q90": float(residual_q90),
            "sector_metrics": _sector_metrics(test, y_test, pred, direction_pred),
            "selected_ridge_alpha": best_alpha,
            "selected_logistic_c": best_c,
            "inner_validation": tuning,
        }
        metrics["horizons"][str(horizon)] = horizon_metrics

        test_predictions[f"actual_abnormal_return_{horizon}d"] = y_test
        test_predictions[f"predicted_abnormal_return_{horizon}d"] = pred
        test_predictions[f"predicted_positive_probability_{horizon}d"] = probability

        # Evaluation uses the chronological holdout. The saved model is then fitted
        # on all available observations so a forecast can use the full history.
        reg_final = make_regressor(best_alpha)
        clf_final = make_classifier(best_c)
        full_target = frame[target].astype(float).to_numpy()
        reg_final.fit(frame[FEATURE_COLUMNS], full_target)
        clf_final.fit(frame[FEATURE_COLUMNS], (full_target > 0).astype(int))
        bundle["horizons"][horizon] = {
            "regressor": reg_final,
            "classifier": clf_final,
            "clip_q01": float(np.quantile(full_target, 0.01)),
            "clip_q99": float(np.quantile(full_target, 0.99)),
            "residual_q10": float(residual_q10),
            "residual_q90": float(residual_q90),
            "evaluation": horizon_metrics,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib", compress=3)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


if __name__ == "__main__":
    output = train_models()
    print(json.dumps(output, ensure_ascii=False, indent=2))
