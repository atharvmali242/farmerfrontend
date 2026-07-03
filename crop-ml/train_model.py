"""
train_model.py
---------------
Trains a crop recommendation model on data/Crop_recommendation.csv and saves:
  - model/crop_model.pkl       (trained Random Forest classifier)
  - model/scaler.pkl           (StandardScaler fit on training features)
  - model/label_encoder.pkl    (LabelEncoder mapping crop names <-> integers)

Run with: python3 train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("data/Crop_recommendation.csv")
print("Loaded dataset:", df.shape[0], "rows,", df.shape[1], "columns")
print("Crops:", sorted(df["label"].unique()))

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 2. Encode crop names to integers (Random Forest needs numeric labels)
# ---------------------------------------------------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ---------------------------------------------------------------------------
# 3. Split into train / test sets (80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------------------------
# 4. Scale features (helps the model treat all 7 inputs fairly —
#    rainfall in mm and ph on a 0-14 scale shouldn't be weighted unevenly)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Train a Random Forest classifier
#    (chosen because it handles non-linear feature interactions well,
#    needs little tuning, and is easy to explain in a viva)
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest accuracy: {:.2%}".format(accuracy))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ---------------------------------------------------------------------------
# 7. Feature importance (useful to show sir which inputs matter most)
# ---------------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importance:\n", importances)

# ---------------------------------------------------------------------------
# 8. Save model, scaler, and encoder for the backend to load
# ---------------------------------------------------------------------------
joblib.dump(model, "model/crop_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")
print("\nSaved model/crop_model.pkl, model/scaler.pkl, model/label_encoder.pkl")
