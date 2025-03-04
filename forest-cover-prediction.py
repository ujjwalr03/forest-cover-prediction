# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 2: Load Dataset (Local CSV)
df = pd.read_csv("train.csv")  # Ensure train.csv is in the same folder

# Step 3: Data Preprocessing
X = df.drop(columns=['Cover_Type'])  # Features
y = df['Cover_Type']  # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save Model
joblib.dump(model, "forest_cover_model.pkl")

# Step 7: Load Model & Predict
loaded_model = joblib.load("forest_cover_model.pkl")
sample_prediction = loaded_model.predict(X_test_scaled[:5])
print("\nSample Predictions:", sample_prediction)

# Step 8: Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1,8), yticklabels=np.arange(1,8))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()