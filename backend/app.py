from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the model and preprocessors
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ridge_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')
encoders_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_encoders.pkl')
imputation_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'imputation_values.pkl')
feature_info_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_info.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)
imputation_values = joblib.load(imputation_path)
feature_info = joblib.load(feature_info_path)

# Load training data for reference
train_df = pd.read_csv('data/train.csv')
mean_price = train_df['SalePrice'].mean()
std_price = train_df['SalePrice'].std()

@app.route('/')
def home():
    return jsonify({
        'message': 'Ames House Price Prediction API',
        'model': 'Ridge Regression',
        'features': len(feature_info['features']),
        'training_samples': len(train_df),
        'avg_price': f"${mean_price:,.0f}",
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'model_info': '/model_info (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Ridge Regression',
        'features': feature_info['features'],
        'numeric_features': feature_info['numeric_cols'],
        'categorical_features': feature_info['categorical_cols'],
        'training_samples': len(train_df),
        'avg_price': mean_price
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        print("Received data:", data)

        # Create DataFrame from input
        input_df = pd.DataFrame([data])

        # Ensure all required features are present
        required_features = feature_info['features']
        numeric_cols = feature_info['numeric_cols']
        categorical_cols = feature_info['categorical_cols']

        # Add missing columns with default values
        for col in required_features:
            if col not in input_df.columns:
                if col in numeric_cols:
                    # Use median from training data
                    input_df[col] = imputation_values['numeric_medians'].get(col, 0)
                else:
                    # Use mode from training data
                    input_df[col] = imputation_values['categorical_modes'].get(col, 'None')

        # Select only required features in correct order
        input_df = input_df[required_features]

        # Handle any remaining missing values
        for col in numeric_cols:
            if col in input_df.columns and pd.isnull(input_df[col]).any():
                input_df[col].fillna(imputation_values['numeric_medians'].get(col, 0), inplace=True)

        for col in categorical_cols:
            if col in input_df.columns and pd.isnull(input_df[col]).any():
                input_df[col].fillna(imputation_values['categorical_modes'].get(col, 'None'), inplace=True)

        # Encode categorical variables
        X_numeric = input_df[numeric_cols].copy()
        X_encoded = X_numeric.copy()

        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    X_encoded[col] = label_encoders[col].transform(input_df[col].astype(str))
                except:
                    # If category not seen in training, use -1
                    X_encoded[col] = -1

        # Ensure correct column order
        X_encoded = X_encoded[feature_info['all_features']]

        # Scale features
        X_scaled = scaler.transform(X_encoded)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Calculate confidence range (based on model's typical error)
        confidence_lower = prediction * 0.9
        confidence_upper = prediction * 1.1

        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'formatted_price': f"${prediction:,.2f}",
            'confidence_range': {
                'lower': round(confidence_lower, 2),
                'upper': round(confidence_upper, 2),
                'formatted_lower': f"${confidence_lower:,.2f}",
                'formatted_upper': f"${confidence_upper:,.2f}"
            },
            'vs_average': {
                'avg_price': f"${mean_price:,.0f}",
                'difference': f"${prediction - mean_price:,.0f}",
                'percent_diff': f"{((prediction - mean_price) / mean_price * 100):.1f}%"
            }
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)