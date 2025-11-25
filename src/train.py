import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data_preprocessing import load_raw_data, preprocess_data
from .config import MODELS_DIR, MLFLOW_DIR

def train_and_log(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        auc = roc_auc_score(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        mlflow.log_param("model", model_name)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(sk_model= pipeline, name = "model")

        return pipeline, auc

print("Model training and logging function defined successfully.")

def main():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    MLFLOW_DIR.mkdir(exist_ok=True, parents=True)

    # Use a proper file URI for Windows paths
    try:
        tracking_uri = MLFLOW_DIR.as_uri()
    except Exception:
        tracking_uri = f"file://{MLFLOW_DIR}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn_prediction")

    df = load_raw_data()
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = []

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("XGBoost", XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2, reg_lambda=2, eval_metric="logloss",random_state=42
        )),
    ]

    for name, model in models:
        pipeline, auc = train_and_log(name, model, X_train, X_test, y_train, y_test, preprocessor)
        candidates.append((auc, name, pipeline))

    # Choose best model
    best_auc, best_name, best_pipeline = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")

    print(f"Best model saved: {best_name} (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    main()
