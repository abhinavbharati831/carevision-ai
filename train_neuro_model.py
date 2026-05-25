import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
parkinsons_data = pd.read_csv('datasets/parkinsons.csv')

# Remove name column
parkinsons_data = parkinsons_data.drop(columns=['name'])

# Features and target
X = parkinsons_data.drop('status', axis=1)
y = parkinsons_data['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Neuro Accuracy: {accuracy * 100:.2f}%')

# Save model
joblib.dump(model, 'models/neuro_model.pkl')

print('Neuro model saved!')