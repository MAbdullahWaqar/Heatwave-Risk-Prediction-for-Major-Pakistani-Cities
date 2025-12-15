from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models(kind="forecast"):
    """
    kind="forecast"  -> climate-only feature set (NO risk_lag_*), best for forward forecasting
    kind="monitoring"-> includes risk_lag_* (better for risk monitoring alerts)
    """
    models = {}

    # Logistic Regression
    models["logreg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # Decision Tree
    models["dtree"] = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    )

    # Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # Gradient Boosting (advanced)
    models["hgb"] = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )

    return models
