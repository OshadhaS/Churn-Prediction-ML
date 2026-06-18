import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("data.csv")  # <-- change path if needed

# 2. Basic cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3. Encoding (same logic you used)
binary_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({
            "Yes": 1, "No": 0,
            "Male": 1, "Female": 0
        })

# 4. Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 5. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model training
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# 8. Save model + scaler
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Training completed and model saved!")