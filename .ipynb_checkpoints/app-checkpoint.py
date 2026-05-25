from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML Models
heart_model = joblib.load('models/heart_model.pkl')
neuro_model = joblib.load('models/neuro_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/neuro')
def neuro():
    return render_template('neuro.html')

# Heart Prediction
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = heart_model.predict(final_features)
        probability = heart_model.predict_proba(final_features)

        confidence = round(np.max(probability) * 100, 2)

        if prediction[0] == 1:
            result = 'High Risk of Heart Disease'
            recommendation = 'Consult a cardiologist immediately.'
        else:
            result = 'Low Risk of Heart Disease'
            recommendation = 'Maintain a healthy lifestyle.'

        return render_template(
            'result.html',
            disease='Heart Disease',
            result=result,
            confidence=confidence,
            recommendation=recommendation
        )

    except Exception as e:
        return str(e)

# Neuro Prediction
@app.route('/predict_neuro', methods=['POST'])
def predict_neuro():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = neuro_model.predict(final_features)

        if prediction[0] == 1:
            result = 'Parkinson’s Indicators Detected'
            recommendation = 'Neurological screening advised.'
            confidence = 92
        else:
            result = 'No Parkinson’s Indicators'
            recommendation = 'No major indicators detected.'
            confidence = 88

        return render_template(
            'result.html',
            disease='Neuro Disease',
            result=result,
            confidence=confidence,
            recommendation=recommendation
        )
    app.run(debug=True)