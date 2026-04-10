import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client

warnings.filterwarnings("ignore")

from app_logger import log

app = Flask(__name__)
# =============================
# Twilio Configuration
# =============================
ACCOUNT_SID = ""
AUTH_TOKEN = ""

TWILIO_NUMBER = ""   # your Twilio number
TO_NUMBER = ""      # your phone number
# =============================
# Load Models
# =============================
pickle_in = bz2.BZ2File('model/classification.pkl', 'rb')
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

# =============================
# Fit Scaler on Training Dataset
# =============================
# Use the dataset available in your project to fit the scaler
df = pd.read_csv("dataset/algerian_forest_fires_dataset_CLEANED.csv")
df.columns = df.columns.str.strip()   # clean column names

# Only keep the 5 features your models were trained on
scaler = StandardScaler()
X = df[['Temperature', 'Ws', 'FFMC', 'DMC', 'ISI']]
scaler.fit(X)
log.info("StandardScaler fitted on [Temperature, Ws, FFMC, DMC, ISI]")

# =============================
# Routes
# =============================
"""def send_alert(message):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=TO_NUMBER
        )
        print("MESSAGE SID:", msg.sid)
    except Exception as e:
        print("TWILIO ERROR:", e)"""
# Homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')
def send_alert(message):
    try:
        print("SENDING SMS...")
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=TO_NUMBER
        )
        print("MESSAGE SID:", msg.sid)
    except Exception as e:
        print("TWILIO ERROR:", str(e))

# API Route for testing with JSON input
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        log.info(f'Input from API: {data}')
        new_data = [list(data.values())]
        new_data_scaled = scaler.transform(new_data)
        output = int(model_C.predict(new_data_scaled)[0])
        if output == 0:
            text = 'Forest is in Safe!'
        else:
            text = 'Forest is Danger!'
        return jsonify(text=text, output=output)
    except Exception as e:
        log.error(f'Error in API input: {e}', exc_info=True)
        return jsonify(error="Check the input again!")


# Classification Model Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['Temperature'])
        ws = float(request.form['Ws'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])

        final_features = [[temperature, ws, ffmc, dmc, isi]]
        final_features_scaled = scaler.transform(final_features)

        output = model_C.predict(final_features_scaled)[0]
        confidence = round(max(model_C.predict_proba(final_features_scaled)[0]) * 100, 2)
        # Get regression risk value
        risk_value = model_R.predict(final_features_scaled)[0]

        # Decide risk level
        if risk_value > 15:
          risk_level = "HIGH"
        elif risk_value > 8:
          risk_level = "MEDIUM"
        else:
         risk_level = "LOW"
        log.info('Prediction done for Classification model')

        if output == 0:
            text = 'Forest is Safe!'
        else:
            text = 'Forest is in Danger!'
            print("ALERT FUNCTION CALLED")   # debug
            alert_msg = f"🔥 WILDFIRE ALERT 🚨\n\nRisk Level: {risk_level}\nTemperature: {temperature}°C\nWind Speed: {ws} km/h\nFire Index: {risk_value:.2f}\n\n⚠️ Take immediate action!"
            send_alert(alert_msg)
        return render_template(
    'index.html',
    prediction_text1=f"{text} | Risk Level: {risk_level} | Index: {risk_value:.2f}",
    confidence=confidence,
     risk_level=risk_level
)
    except Exception as e:
        log.error(f'Input error in Classification: {e}', exc_info=True)
        return render_template('index.html', prediction_text1="Check the Input again!!!")

# Regression Model Route
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        temperature = float(request.form['Temperature'])
        ws = float(request.form['Ws'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])

        data = [[temperature, ws, ffmc, dmc, isi]]
        data_scaled = scaler.transform(data)

        output = model_R.predict(data_scaled)[0]
        log.info('Prediction done for Regression model')

        if output > 15:
            send_alert(f"🔥 HIGH FIRE RISK!\nIndex: {output:.2f}\nTake action immediately!")
            return render_template(
                'index.html',
                prediction_text2=f"Fuel Moisture Code index = {output:.4f} ---- Warning!!! High hazard rating"
            )
        else:
            return render_template(
                'index.html',
                prediction_text2=f"Fuel Moisture Code index = {output:.4f} ---- Safe.. Low hazard rating"
            )

    except Exception as e:
        log.error(f'Input error in Regression: {e}', exc_info=True)
        return render_template('index.html', prediction_text2="Check the Input again!!!")


# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=False)
