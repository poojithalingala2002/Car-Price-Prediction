from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained objects
model = joblib.load('reg_model.pkl')
scaler = joblib.load('scaler.pkl')
te = joblib.load('target_encode.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])

        Fuel_Type = request.form['Fuel_Type']
        Seller_Type = request.form['Seller_Type']
        Transmission = request.form['Transmission']
        Brand = request.form['Brand']

        # ---- Mapping ----
        Fuel_Type = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[Fuel_Type]
        Seller_Type = {'Individual': 0, 'Dealer': 1}[Seller_Type]
        Transmission = {'Manual': 0, 'Automatic': 1}[Transmission]

        input_df = pd.DataFrame({
            'Year': [Year],
            'Present_Price': [Present_Price],
            'Kms_Driven': [Kms_Driven],
            'Owner': [Owner],
            'Fuel_Type': [Fuel_Type],
            'Seller_Type': [Seller_Type],
            'Transmission': [Transmission],
            'Brand': [Brand]
        })

        # Normalize Brand
        input_df['Brand'] = input_df['Brand'].str.strip().str.title()

        # Target Encode
        input_df['Brand'] = te.transform(input_df[['Brand']])

        # Column order (CRITICAL)
        final_cols = [
            'Year',
            'Present_Price',
            'Kms_Driven',
            'Owner',
            'Fuel_Type',
            'Seller_Type',
            'Transmission',
            'Brand'
        ]
        input_df = input_df[final_cols]

        # Scale selected columns
        scale_cols = ['Year', 'Present_Price', 'Kms_Driven', 'Brand']
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        # Predict
        prediction = model.predict(input_df)[0]

        # If target was log-transformed
        # prediction = np.expm1(prediction)

        prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
