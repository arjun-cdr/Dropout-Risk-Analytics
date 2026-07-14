import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_excel("prediction_dataset.xlsx")

target = "DroppedOut"
X = df.drop(columns=[target])
y = df[target]

# ---------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------
label_encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# ---------------------------
# 2. TRAIN / TEST / VALIDATION SPLIT
# 70% train, 15% test, 15% validation
# ---------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ---------------------------
# 3. TRAIN RANDOM FOREST MODEL
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------------------
# TEST + VALIDATION PERFORMANCE
# ---------------------------
y_pred_test = model.predict(X_test)
y_pred_val = model.predict(X_val)

print("TEST ACCURACY:", accuracy_score(y_test, y_pred_test))
print("TEST REPORT:\n", classification_report(y_test, y_pred_test))

print("VALIDATION ACCURACY:", accuracy_score(y_val, y_pred_val))
print("VALIDATION REPORT:\n", classification_report(y_val, y_pred_val))

# ---------------------------
# 4. FUTURE DROPOUT TREND PREDICTION
# ---------------------------
# Get dropout probability for every student
dropout_probs = model.predict_proba(X)[:, 1]

# Average dropout risk today
current_risk = dropout_probs.mean()

# FUTURE PROJECTION (simple model)
# years = np.arange(2025, 2027)  # next 6 years
# future_risk = current_risk * (1 + np.linspace(0.02, 0.20, len(years)))  # assume 2–20% increase

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(years, future_risk * 100, marker='o')
# plt.title("Projected Dropout Percentage (Next 2-3 Years)")
# plt.xlabel("Year")
# plt.ylabel("Predicted Dropout %")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ---------------------------
# Dropout Probability Distribution
# ---------------------------
dropout_probs = model.predict_proba(X)[:, 1]

plt.figure(figsize=(8, 5))
plt.hist(dropout_probs, bins=10, edgecolor='black')
plt.title("Dropout Probability Distribution")
plt.xlabel("Dropout Probability")
plt.ylabel("Number of Students")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()