import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import NUMERIC, CATEGORICAL, FEATURES

df = pd.read_csv("data/housing_processed.csv")
X = df[FEATURES]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("real-estate-investment-advisor")


def make_pipeline(model, num_cols=NUMERIC, cat_cols=CATEGORICAL):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def top_features(pipe, n=20):
    names = pipe.named_steps["prep"].get_feature_names_out()
    m = pipe.named_steps["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(np.ravel(m.coef_))
    return pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False).head(n)


# ======================= CLASSIFICATION: Good_Investment =======================
y = df["Good_Investment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=-1),
}
clf_results = {}
for name, model in clf_models.items():
    with mlflow.start_run(run_name=f"clf_{name}") as run:
        pipe = make_pipeline(model).fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        m = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
             "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba)}
        mlflow.set_tag("task", "classification")
        mlflow.log_param("model", name)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        clf_results[name] = {"metrics": m, "pipe": pipe, "run_id": run.info.run_id}
        print(f"[clf] {name:20s} accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}")

# extra run: the experiment you just did, logged for your report
with mlflow.start_run(run_name="clf_RF_without_price_columns"):
    hidden = ["Price_vs_City_Median", "Price_per_SqFt", "Price_in_Lakhs"]
    pipe = make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
                         [c for c in NUMERIC if c not in hidden]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mlflow.set_tag("task", "classification-experiment")
    mlflow.log_metrics({"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)})

best_clf = max([n for n in clf_results if n != "Baseline"], key=lambda n: clf_results[n]["metrics"]["f1"])

# ======================= REGRESSION: Future_Price_Lakhs =======================
y = df["Future_Price_Lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_models = {
    "Baseline": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.08, n_jobs=-1),
}
reg_results = {}
for name, model in reg_models.items():
    with mlflow.start_run(run_name=f"reg_{name}") as run:
        pipe = make_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = {"rmse": float(np.sqrt(mean_squared_error(y_test, pred))), "mae": mean_absolute_error(y_test, pred),
             "r2": r2_score(y_test, pred)}
        mlflow.set_tag("task", "regression")
        mlflow.log_param("model", name)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        reg_results[name] = {"metrics": m, "pipe": pipe, "run_id": run.info.run_id}
        print(f"[reg] {name:20s} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.4f}")

best_reg = min([n for n in reg_results if n != "Baseline"], key=lambda n: reg_results[n]["metrics"]["rmse"])

# ======================= SAVE best models + extras for the app =======================
joblib.dump(clf_results[best_clf]["pipe"], "models/classifier.joblib")
joblib.dump(reg_results[best_reg]["pipe"], "models/regressor.joblib")
top_features(clf_results[best_clf]["pipe"]).to_csv("models/importance_clf.csv", index=False)
top_features(reg_results[best_reg]["pipe"]).to_csv("models/importance_reg.csv", index=False)
mlflow.register_model(f"runs:/{clf_results[best_clf]['run_id']}/model", "GoodInvestmentClassifier")
mlflow.register_model(f"runs:/{reg_results[best_reg]['run_id']}/model", "FuturePriceRegressor")

options = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}
options["City_to_State"] = df.groupby("City")["State"].first().to_dict()
options["amenities"] = sorted({a.strip() for s in df["Amenities"] for a in s.split(",")})
json.dump(options, open("models/options.json", "w"), indent=2)
json.dump({"best_classifier": best_clf, "best_regressor": best_reg,
           "classification": {n: r["metrics"] for n, r in clf_results.items()},
           "regression": {n: r["metrics"] for n, r in reg_results.items()}},
          open("models/metrics.json", "w"), indent=2)
print("\nBest classifier:", best_clf, "| best regressor:", best_reg)