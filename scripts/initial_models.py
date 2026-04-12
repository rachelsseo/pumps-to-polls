"""
IERAD Predictive Model — Emergency Resource Allocation & Routing
----------------------------------------------------------------
Two models:
  1. Dispatch classifier  → recommends Label (Ambulance Only / Drone Only / Hybrid)
  2. Response time regressor → estimates response time for route prioritization

Usage:
  python model.py            # train + evaluate both models
  python model.py --predict  # demo real-time recommendation
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & join ─────────────────────────────────────────────────────────────

def load_data():
    incident     = pd.read_parquet("../parquet-data/incident.parquet")
    env          = pd.read_parquet("../parquet-data/environmental_conditions.parquet")
    resource     = pd.read_parquet("../parquet-data/resource.parquet")
    dispatch_df  = pd.read_parquet("../parquet-data/dispatch.parquet")

    df = (incident
          .merge(env.drop(columns="condition_id"), on="incident_id")
          .merge(resource.drop(columns="resource_id"), on="incident_id")
          .merge(dispatch_df.drop(columns="dispatch_id"), on="incident_id"))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"]      = df["Timestamp"].dt.hour
    df["weekday"]   = df["Timestamp"].dt.weekday
    return df

# ── 2. Feature config ──────────────────────────────────────────────────────────

CAT_FEATURES = [
    "Incident_Type", "Incident_Severity", "Emergency_Level",
    "Region_Type", "Road_Type",
    "Weather_Condition", "Weather_Impact", "Traffic_Congestion", "Air_Traffic",
    "Drone_Availability", "Ambulance_Availability", "Specialist_Availability",
    "Dispatch_Coordinator",
]

NUM_FEATURES = [
    "Number_of_Injuries", "Distance_to_Incident", "Battery_Life",
    "Drone_Speed", "Ambulance_Speed", "Payload_Weight", "Fuel_Level",
    "Hospital_Capacity", "hour", "weekday",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
DISPATCH_TARGET   = "Label"
RESPONSE_TARGET   = "Response_Time"

# ── 3. Preprocessing pipeline ─────────────────────────────────────────────────

def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])

# ── 4. Train dispatch classifier ──────────────────────────────────────────────

def train_dispatch_model(df):
    X = df[ALL_FEATURES]
    y = df[DISPATCH_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n── Dispatch Classifier ──────────────────────────────────")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "dispatch_model.pkl")
    return model

# ── 5. Train response time regressor ──────────────────────────────────────────

def train_response_model(df):
    X = df[ALL_FEATURES]
    y = df[RESPONSE_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # n_estimators=200 for better performance, but reduced to 50 for faster analysis
    model = Pipeline([
        ("pre", build_preprocessor()),
        ("reg", GradientBoostingRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.05, random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print("\n── Response Time Regressor ──────────────────────────────")
    print(f"  MAE:  {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  R²:   {r2:.3f}")

    joblib.dump(model, "response_model.pkl")
    return model

# ── 6. Feature importance ──────────────────────────────────────────────────────

def print_top_features(model, n=10):
    pre   = model.named_steps["pre"]
    clf   = model.named_steps.get("clf") or model.named_steps.get("reg")
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    feat_names = np.concatenate([cat_names, NUM_FEATURES])
    importances = pd.Series(clf.feature_importances_, index=feat_names)
    print(f"\n  Top {n} features:")
    print(importances.nlargest(n).to_string())

# ── 7. Real-time recommendation ───────────────────────────────────────────────

def recommend(incident_data: dict, dispatch_model, response_model) -> dict:
    """
    Given a live incident dict, return dispatch recommendation + ETA.

    incident_data keys must match ALL_FEATURES. If Timestamp fields
    (hour, weekday) aren't pre-computed, pass 'Timestamp' as ISO string
    and they'll be derived.
    """
    if "Timestamp" in incident_data and "hour" not in incident_data:
        ts = pd.to_datetime(incident_data.pop("Timestamp"))
        incident_data["hour"]    = ts.hour
        incident_data["weekday"] = ts.weekday()

    row = pd.DataFrame([incident_data])

    dispatch_label = dispatch_model.predict(row)[0]
    dispatch_probs = dispatch_model.predict_proba(row)[0]
    classes        = dispatch_model.classes_
    confidence     = dict(zip(classes, (dispatch_probs * 100).round(1)))

    estimated_time = response_model.predict(row)[0]

    return {
        "recommended_dispatch": dispatch_label,
        "confidence_%":         confidence,
        "estimated_response_min": round(estimated_time, 1),
    }

# ── 8. Demo incident ───────────────────────────────────────────────────────────

DEMO_INCIDENT = {
    "Incident_Type":          "Cardiac Arrest",
    "Incident_Severity":      "High",
    "Emergency_Level":        "Critical",
    "Region_Type":            "Urban",
    "Road_Type":              "Highway",
    "Weather_Condition":      "Clear",
    "Weather_Impact":         "None",
    "Traffic_Congestion":     "High",
    "Air_Traffic":            "Low",
    "Drone_Availability":     "Available",
    "Ambulance_Availability": "Available",
    "Specialist_Availability":"Available",
    "Dispatch_Coordinator":   "AI",
    "Number_of_Injuries":     2,
    "Distance_to_Incident":   12.5,
    "Battery_Life":           85.0,
    "Drone_Speed":            72.0,
    "Ambulance_Speed":        55.0,
    "Payload_Weight":         5.0,
    "Fuel_Level":             78.0,
    "Hospital_Capacity":      60,
    "hour":                   14,
    "weekday":                1,
}

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Run demo real-time recommendation only")
    args = parser.parse_args()

    if args.predict:
        dispatch_model = joblib.load("dispatch_model.pkl")
        response_model = joblib.load("response_model.pkl")
    else:
        print("Loading data...")
        df = load_data()
        print(f"  {len(df):,} incidents loaded, {df[DISPATCH_TARGET].value_counts().to_dict()}")

        dispatch_model = train_dispatch_model(df)
        print_top_features(dispatch_model)

        response_model = train_response_model(df)
        print_top_features(response_model)

        print("\nModels saved: dispatch_model.pkl, response_model.pkl")

    print("\n── Demo recommendation ──────────────────────────────────")
    result = recommend(DEMO_INCIDENT.copy(), dispatch_model, response_model)
    for k, v in result.items():
        print(f"  {k}: {v}")