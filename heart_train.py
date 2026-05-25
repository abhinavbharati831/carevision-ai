import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
data = pd.read_csv('data/heart.csv')

# Features and Target
X = data.drop('target', axis=1)
y = data['target']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier()

# Train Model
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Heart Model Accuracy:", acc)

# Save Model
joblib.dump(model, 'saved_models/heart_model.pkl')

print("Heart Model Saved")