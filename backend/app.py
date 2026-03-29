from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load the model and preprocessors
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ridge_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')
encoders_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_encoders.pkl')
imputation_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'imputation_values.pkl')
feature_info_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_info.pkl')

# Load training data for visualizations
train_df = pd.read_csv('data/train.csv')

# Load models
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    imputation_values = joblib.load(imputation_path)
    feature_info = joblib.load(feature_info_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please run 'python model.py' first to train the model.")
    exit(1)

# Calculate statistics for visualizations
mean_price = train_df['SalePrice'].mean()
median_price = train_df['SalePrice'].median()
std_price = train_df['SalePrice'].std()
min_price = train_df['SalePrice'].min()
max_price = train_df['SalePrice'].max()

@app.route('/')
def home():
    return jsonify({
        'message': 'Ames House Price Prediction API',
        'model': 'Ridge Regression',
        'features': len(feature_info['features']) if feature_info else 22,
        'training_samples': len(train_df),
        'avg_price': f"${mean_price:,.0f}",
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get basic statistics about the dataset"""
    return jsonify({
        'mean_price': float(mean_price),
        'median_price': float(median_price),
        'std_price': float(std_price),
        'min_price': float(min_price),
        'max_price': float(max_price),
        'total_houses': len(train_df),
        'avg_living_area': float(train_df['GrLivArea'].mean()),
        'avg_lot_area': float(train_df['LotArea'].mean()),
        'avg_qual_rating': float(train_df['OverallQual'].mean()),
        'avg_garage_cars': float(train_df['GarageCars'].mean())
    })

@app.route('/api/price-distribution', methods=['GET'])
def get_price_distribution():
    """Get price distribution data for histogram"""
    prices = train_df['SalePrice'].tolist()
    return jsonify({
        'prices': prices,
        'bins': 50
    })

@app.route('/api/feature-correlation', methods=['GET'])
def get_feature_correlation():
    """Get correlation matrix for key features"""
    # Select key features for correlation
    key_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'LotArea',
                    'OverallQual', 'YearBuilt', 'GarageCars', 'FullBath']

    correlation_matrix = train_df[key_features].corr().round(2)

    # Convert to list of lists for JSON
    correlation_data = {
        'features': key_features,
        'matrix': correlation_matrix.values.tolist()
    }

    return jsonify(correlation_data)

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance based on model coefficients"""
    # Get coefficients from trained model
    coefficients = model.coef_
    feature_names = feature_info['all_features']

    # Get top 15 features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coefficients)
    }).sort_values('importance', ascending=False).head(15)

    return jsonify({
        'features': importance_df['feature'].tolist(),
        'importance': importance_df['importance'].tolist()
    })

@app.route('/api/price-by-neighborhood', methods=['GET'])
def get_price_by_neighborhood():
    """Get average price by neighborhood"""
    neighborhood_prices = train_df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count']).reset_index()
    neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=False)

    return jsonify({
        'neighborhoods': neighborhood_prices['Neighborhood'].tolist(),
        'prices': neighborhood_prices['mean'].tolist(),
        'counts': neighborhood_prices['count'].tolist()
    })

@app.route('/api/price-over-time', methods=['GET'])
def get_price_over_time():
    """Get average price by year built"""
    yearly_prices = train_df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
    yearly_counts = train_df.groupby('YearBuilt').size().reset_index(name='count')

    return jsonify({
        'years': yearly_prices['YearBuilt'].tolist(),
        'prices': yearly_prices['SalePrice'].tolist(),
        'counts': yearly_counts['count'].tolist()
    })

@app.route('/api/price-by-quality', methods=['GET'])
def get_price_by_quality():
    """Get price distribution by overall quality"""
    quality_stats = train_df.groupby('OverallQual')['SalePrice'].agg(['mean', 'min', 'max', 'count']).reset_index()

    return jsonify({
        'qualities': quality_stats['OverallQual'].tolist(),
        'means': quality_stats['mean'].tolist(),
        'mins': quality_stats['min'].tolist(),
        'maxs': quality_stats['max'].tolist(),
        'counts': quality_stats['count'].tolist()
    })

@app.route('/api/scatter-data', methods=['GET'])
def get_scatter_data():
    """Get data for scatter plots"""
    feature = request.args.get('feature', 'GrLivArea')

    if feature not in train_df.columns:
        return jsonify({'error': 'Feature not found'}), 400

    data = train_df[[feature, 'SalePrice']].dropna()

    return jsonify({
        'x': data[feature].tolist(),
        'y': data['SalePrice'].tolist(),
        'feature': feature
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

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
                    input_df[col] = imputation_values['numeric_medians'].get(col, 0)
                else:
                    input_df[col] = imputation_values['categorical_modes'].get(col, 'None')

        # Select only required features
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
                    X_encoded[col] = -1

        # Ensure correct column order
        X_encoded = X_encoded[feature_info['all_features']]

        # Scale features
        X_scaled = scaler.transform(X_encoded)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Calculate confidence range
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