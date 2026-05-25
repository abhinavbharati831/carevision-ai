import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
heart_data = pd.read_csv('datasets/heart.csv')

# Features and target
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Heart Disease Accuracy: {accuracy * 100:.2f}%')

# Save Model
joblib.dump(model, 'models/heart_model.pkl')

print('Heart model saved!')