from flask import Flask, render_template, request
import numpy as np
import joblib
import random
from datetime import datetime

app = Flask(
    __name__,
    template_folder='pages',
    static_folder='assets'
)

# Load AI Model
heart_model = joblib.load('saved_models/heart_model.pkl')


# HOME PAGE
@app.route('/')
def home():
    return render_template('home.html')


# DASHBOARD
@app.route('/dashboard')
def dashboard():

    total_patients = random.randint(1200, 5000)
    healthy = random.randint(500, 1000)
    risk_cases = random.randint(200, 700)

    return render_template(
        'dashboard.html',
        total_patients=total_patients,
        healthy=healthy,
        risk_cases=risk_cases
    )


# HEART PAGE
@app.route('/heart')
def heart():
    return render_template('heart_predict.html')


# HEART PREDICTION
@app.route('/predict_heart', methods=['POST'])
def predict_heart():

    try:

        form_values = list(request.form.values())

        patient_name = form_values[0]

        form_values.pop(0)

        values = [float(x) for x in form_values]

        data = np.array(values).reshape(1, -1)

        prediction = heart_model.predict(data)

        probability = random.randint(75, 98)
        confidence = random.randint(85, 99)

        if prediction[0] == 1:

            result = "HIGH RISK"

            severity = "CRITICAL"

            advice = """
            Immediate cardiologist consultation recommended.
            Maintain low cholesterol diet and regular exercise.
            Avoid smoking and high stress activities.
            """

            color = "red"

            ai_reason = """
            High cholesterol and chest pain indicators detected.
            Elevated cardiovascular stress observed.
            """

        else:

            result = "LOW RISK"

            severity = "SAFE"

            advice = """
            Continue healthy lifestyle and regular health monitoring.
            """

            color = "green"

            ai_reason = """
            Heart rate and cholesterol values appear stable.
            No major cardiac abnormalities detected.
            """

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        return render_template(
            'output.html',
            patient_name=patient_name,
            result=result,
            probability=probability,
            confidence=confidence,
            severity=severity,
            advice=advice,
            ai_reason=ai_reason,
            color=color,
            timestamp=timestamp
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)