import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("feature_engineered_dataset.csv")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

features = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls",
    "total_charges",
    "tenure_years",
    "support_call_rate",
    "age_group_encoded"
]

X = df[features]
y = df["churn"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE MODEL PIPELINES
# ==========================================

models = {

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        ))
    ])
}


# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

for name, pipeline in models.items():

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Training Time": training_time
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1-Score :", round(f1_score(y_test, y_pred, zero_division=0), 4))


# ==========================================
# 6. MODEL LEADERBOARD
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\n========================================")
print("MODEL LEADERBOARD")
print("========================================")
print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ==========================================
# 7. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\n========================================")
print("BEST RANDOM FOREST PARAMETERS")
print("========================================")
print(grid_search.best_params_)
print("Best CV F1:", round(grid_search.best_score_, 4))


# ==========================================
# 8. XGBOOST HYPERPARAMETER TUNING
# ==========================================

xgb_pipeline = Pipeline([
    ("model", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

print("\n========================================")
print("BEST XGBOOST PARAMETERS")
print("========================================")
print(xgb_grid.best_params_)
print("Best CV F1:", round(xgb_grid.best_score_, 4))


# ==========================================
# 9. EVALUATE TUNED MODELS
# ==========================================

tuned_models = {
    "Tuned Random Forest": best_rf,
    "Tuned XGBoost": best_xgb
}

tuned_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)

    tuned_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, zero_division=0
        )
    })


tuned_df = pd.DataFrame(tuned_results)

print("\n========================================")
print("TUNED MODEL RESULTS")
print("========================================")
print(tuned_df)

tuned_df.to_csv(
    "tuned_model_comparison.csv",
    index=False
)


# ==========================================
# 10. SELECT BEST MODEL
# ==========================================

all_results = pd.concat(
    [
        results_df[
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        ],
        tuned_df
    ],
    ignore_index=True
)

best_row = all_results.loc[
    all_results["F1-Score"].idxmax()
]

best_model_name = best_row["Model"]

if best_model_name == "Tuned Random Forest":
    final_model = best_rf

elif best_model_name == "Tuned XGBoost":
    final_model = best_xgb

else:
    final_model = trained_models[best_model_name]


print("\n========================================")
print("BEST MODEL")
print("========================================")
print("Model:", best_model_name)
print("Accuracy:", round(best_row["Accuracy"], 4))
print("Precision:", round(best_row["Precision"], 4))
print("Recall:", round(best_row["Recall"], 4))
print("F1-Score:", round(best_row["F1-Score"], 4))


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

final_predictions = final_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    final_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churn", "Churn"],
    yticklabels=["Not Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. FEATURE IMPORTANCE
# ==========================================

if best_model_name in [
    "Random Forest",
    "Tuned Random Forest",
    "XGBoost",
    "Tuned XGBoost"
]:

    importances = final_model.named_steps[
        "model"
    ].feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")
    print(feature_importance_df)

    feature_importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - {best_model_name}")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

joblib.dump(
    final_model,
    "best_churn_model.pkl"
)

print("\nBest model saved as: best_churn_model.pkl")


# ==========================================
# 14. SAVE FINAL EVALUATION REPORT DATA
# ==========================================

final_report = pd.DataFrame([{
    "Best Model": best_model_name,
    "Accuracy": best_row["Accuracy"],
    "Precision": best_row["Precision"],
    "Recall": best_row["Recall"],
    "F1-Score": best_row["F1-Score"]
}])

final_report.to_csv(
    "final_evaluation.csv",
    index=False
)


# ==========================================
# 15. PROJECT COMPLETED
# ==========================================

print("\n========================================")
print("DAY 55 PROJECT COMPLETED")
print("========================================")
print("Generated files:")
print("- model_comparison.csv")
print("- tuned_model_comparison.csv")
print("- final_evaluation.csv")
print("- confusion_matrix.png")
print("- feature_importance.csv")
print("- feature_importance.png")
print("- best_churn_model.pkl")