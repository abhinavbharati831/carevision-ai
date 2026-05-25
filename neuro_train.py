import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
data = pd.read_csv('data/parkinsons.csv')

# Remove Name Column
data = data.drop(columns=['name'])

# Select ONLY required columns
X = data[[
    'MDVP:Fo(Hz)',
    'MDVP:Fhi(Hz)',
    'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Shimmer',
    'HNR',
    'RPDE',
    'DFA'
]]

# Target
y = data['status']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = SVC()

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Neuro Model Accuracy:", acc)

# Save Model
joblib.dump(model, 'saved_models/neuro_model.pkl')

print("Neuro Model Saved")