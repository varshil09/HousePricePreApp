# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Loading Ames Housing Dataset...")
print("="*60)

# Load the data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# ============================================
# STEP 1: Select Most Important Features
# ============================================
print("\n" + "="*60)
print("STEP 1: Selecting Key Features")
print("="*60)

# Based on domain knowledge and common sense, these are the most important features
key_features = [
    # Size features
    'GrLivArea',           # Above grade living area
    'TotalBsmtSF',         # Total basement area
    'LotArea',             # Lot size
    '1stFlrSF',            # First floor area
    '2ndFlrSF',            # Second floor area

    # Quality features
    'OverallQual',         # Overall material and finish quality
    'OverallCond',         # Overall condition
    'YearBuilt',           # Original construction date
    'YearRemodAdd',        # Remodel date

    # Room features
    'BedroomAbvGr',        # Bedrooms above grade
    'KitchenAbvGr',        # Kitchens above grade
    'TotRmsAbvGrd',        # Total rooms above grade
    'FullBath',            # Full bathrooms
    'HalfBath',            # Half bathrooms

    # Garage features
    'GarageCars',          # Garage size in cars
    'GarageArea',          # Garage area

    # Location/Neighborhood (categorical)
    'Neighborhood',        # Physical location
    'MSZoning',            # Zoning classification

    # Other important features
    'Fireplaces',          # Number of fireplaces
    'WoodDeckSF',          # Wood deck area
    'OpenPorchSF',         # Open porch area
    'LotFrontage',         # Street frontage
]

print(f"Selected {len(key_features)} key features for modeling")

# Separate features and target from training data
X_train_full = train_df[key_features].copy()
y_train_full = train_df['SalePrice']

# Test data (no SalePrice column)
X_test = test_df[key_features].copy()

print(f"\nFeatures selected:")
for i, feat in enumerate(key_features, 1):
    print(f"  {i}. {feat}")

# ============================================
# STEP 2: Handle Missing Values
# ============================================
print("\n" + "="*60)
print("STEP 2: Handling Missing Values")
print("="*60)

# Check missing values in training data
train_missing = X_train_full.isnull().sum()
train_missing_cols = train_missing[train_missing > 0]

if len(train_missing_cols) > 0:
    print("\nTraining data - columns with missing values:")
    for col, count in train_missing_cols.items():
        pct = (count / len(X_train_full)) * 100
        print(f"  {col}: {count} missing ({pct:.1f}%)")
else:
    print("\nNo missing values in training data!")

# Check missing values in test data
test_missing = X_test.isnull().sum()
test_missing_cols = test_missing[test_missing > 0]

if len(test_missing_cols) > 0:
    print("\nTest data - columns with missing values:")
    for col, count in test_missing_cols.items():
        pct = (count / len(X_test)) * 100
        print(f"  {col}: {count} missing ({pct:.1f}%)")

# Separate numeric and categorical columns
numeric_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Create copies to work with
X_train_processed = X_train_full.copy()
X_test_processed = X_test.copy()

# Store median values for numeric columns (to use on test data)
numeric_medians = {}

# Handle missing values in numeric columns (training data)
print("\nHandling numeric missing values in training data...")
for col in numeric_cols:
    if X_train_processed[col].isnull().any():
        median_val = X_train_processed[col].median()
        numeric_medians[col] = median_val
        X_train_processed[col].fillna(median_val, inplace=True)
        print(f"  {col}: filled with median ({median_val:.2f})")
    else:
        numeric_medians[col] = X_train_processed[col].median()

# Handle missing values in categorical columns (training data)
print("\nHandling categorical missing values in training data...")
categorical_modes = {}
for col in categorical_cols:
    if X_train_processed[col].isnull().any():
        mode_val = X_train_processed[col].mode()[0]
        categorical_modes[col] = mode_val
        X_train_processed[col].fillna(mode_val, inplace=True)
        print(f"  {col}: filled with mode ({mode_val})")
    else:
        categorical_modes[col] = X_train_processed[col].mode()[0]

# Apply same imputation to test data
print("\nApplying same imputation to test data...")
for col in numeric_cols:
    if col in X_test_processed.columns and X_test_processed[col].isnull().any():
        X_test_processed[col].fillna(numeric_medians[col], inplace=True)
        print(f"  {col}: filled with median ({numeric_medians[col]:.2f})")

for col in categorical_cols:
    if col in X_test_processed.columns and X_test_processed[col].isnull().any():
        X_test_processed[col].fillna(categorical_modes[col], inplace=True)
        print(f"  {col}: filled with mode ({categorical_modes[col]})")

# Verify no missing values remain
print(f"\nRemaining missing values in training: {X_train_processed.isnull().sum().sum()}")
print(f"Remaining missing values in test: {X_test_processed.isnull().sum().sum()}")

# ============================================
# STEP 3: Encode Categorical Variables
# ============================================
print("\n" + "="*60)
print("STEP 3: Encoding Categorical Variables")
print("="*60)

# Create dataframes for encoded features
X_train_encoded = X_train_processed[numeric_cols].copy()
X_test_encoded = X_test_processed[numeric_cols].copy()

# Apply Label Encoding to categorical columns
label_encoders = {}

for col in categorical_cols:
    print(f"Encoding: {col}")

    # Combine training and test data to see all possible categories
    all_values = pd.concat([
        X_train_processed[col].astype(str),
        X_test_processed[col].astype(str)
    ]).unique()

    le = LabelEncoder()
    le.fit(all_values)

    # Transform training data
    X_train_encoded[col] = le.transform(X_train_processed[col].astype(str))

    # Transform test data (handle unseen categories)
    try:
        X_test_encoded[col] = le.transform(X_test_processed[col].astype(str))
    except:
        # For any unseen categories, use -1 or 0
        X_test_encoded[col] = X_test_processed[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    label_encoders[col] = le

print(f"\nFinal training feature matrix shape: {X_train_encoded.shape}")
print(f"Final test feature matrix shape: {X_test_encoded.shape}")

# ============================================
# STEP 4: Scale the Features
# ============================================
print("\n" + "="*60)
print("STEP 4: Scaling Features")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print(f"Training features scaled successfully!")
print(f"Test features scaled successfully!")

# ============================================
# STEP 5: Train Ridge Regression with Hyperparameter Tuning
# ============================================
print("\n" + "="*60)
print("STEP 5: Training Ridge Regression Model")
print("="*60)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train_full, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples (for final submission)")

# Try different alpha values to find the best
alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
best_alpha = None
best_score = -np.inf
best_model = None
results = []

print("\nTesting different alpha values...")
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    results.append({'alpha': alpha, 'r2': r2, 'rmse': rmse})
    print(f"  Alpha={alpha:6.2f}: R² = {r2:.4f}, RMSE = ${rmse:,.2f}")

    if r2 > best_score:
        best_score = r2
        best_alpha = alpha
        best_model = ridge

print(f"\n✅ Best alpha: {best_alpha} with validation R² = {best_score:.4f}")

# Train final model with best alpha on ALL training data
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train_full)

# ============================================
# STEP 6: Evaluate on Validation Set
# ============================================
print("\n" + "="*60)
print("STEP 6: Final Model Evaluation")
print("="*60)

# Predict on validation set
y_val_pred = final_model.predict(X_val)

# Calculate metrics
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print("\n" + "-"*40)
print("FINAL MODEL PERFORMANCE (on validation set)")
print("-"*40)
print(f"R² Score:  {val_r2:.4f}")
print(f"RMSE:      ${val_rmse:,.2f}")
print(f"MAE:       ${val_mae:,.2f}")
print(f"Mean Price: ${y_train_full.mean():,.2f}")
print(f"Error %:    {(val_mae / y_train_full.mean() * 100):.1f}%")

# Cross-validation score
cv_scores = cross_val_score(final_model, X_train_scaled, y_train_full, cv=5, scoring='r2')
print(f"\n5-Fold Cross-Validation R² Scores:")
print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================
# STEP 7: Make Predictions on Test Data
# ============================================
print("\n" + "="*60)
print("STEP 7: Making Predictions on Test Data")
print("="*60)

# Predict on test data
test_predictions = final_model.predict(X_test_scaled)

# Create submission dataframe
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

print(f"\nGenerated {len(submission)} predictions for test data")
print("\nSample predictions (first 10):")
print(submission.head(10).to_string(index=False))

# Save submission file
submission.to_csv('data/submission.csv', index=False)
print("\n✅ Submission file saved as 'data/submission.csv'")

# ============================================
# STEP 8: Feature Importance Analysis
# ============================================
print("\n" + "="*60)
print("STEP 8: Feature Importance Analysis")
print("="*60)

# Get coefficients
coefficients = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'coefficient': final_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features (by coefficient magnitude):")
print(coefficients.head(10).to_string(index=False))

print("\nTop 5 Positive Influences (increase price):")
print(coefficients[coefficients['coefficient'] > 0].head(5).to_string(index=False))

print("\nTop 5 Negative Influences (decrease price):")
print(coefficients[coefficients['coefficient'] < 0].head(5).to_string(index=False))

# ============================================
# STEP 9: Save the Model and Preprocessors
# ============================================
print("\n" + "="*60)
print("STEP 9: Saving Model and Preprocessors")
print("="*60)

# Save the model
joblib.dump(final_model, '../model/ridge_model.pkl')

# Save the scaler
joblib.dump(scaler, '../model/scaler.pkl')

# Save label encoders
joblib.dump(label_encoders, '../model/label_encoders.pkl')

# Save imputation values
imputation_values = {
    'numeric_medians': numeric_medians,
    'categorical_modes': categorical_modes
}
joblib.dump(imputation_values, '../model/imputation_values.pkl')

# Save feature info
feature_info = {
    'features': key_features,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'all_features': X_train_encoded.columns.tolist()
}
joblib.dump(feature_info, '../model/feature_info.pkl')

print("✓ Ridge model saved as '../model/ridge_model.pkl'")
print("✓ Scaler saved as '../model/scaler.pkl'")
print("✓ Label encoders saved as '../model/label_encoders.pkl'")
print("✓ Imputation values saved as '../model/imputation_values.pkl'")
print("✓ Feature info saved as '../model/feature_info.pkl'")

# ============================================
# STEP 10: Detailed Error Analysis
# ============================================
print("\n" + "="*60)
print("STEP 10: Error Analysis")
print("="*60)

# Calculate errors on validation set
errors = y_val - y_val_pred
abs_errors = np.abs(errors)
pct_errors = (abs_errors / y_val) * 100

print(f"\nError Statistics:")
print(f"  Mean Error:        ${errors.mean():,.2f}")
print(f"  Mean Absolute Error: ${abs_errors.mean():,.2f}")
print(f"  Median Absolute Error: ${np.median(abs_errors):,.2f}")
print(f"  Mean % Error:      {pct_errors.mean():.1f}%")
print(f"  Median % Error:    {np.median(pct_errors):.1f}%")
print(f"  Max Error:         ${abs_errors.max():,.2f}")
print(f"  Min Error:         ${abs_errors.min():,.2f}")

# Error distribution
print(f"\nError Distribution:")
print(f"  Within 5%:    {(pct_errors <= 5).sum() / len(pct_errors) * 100:.1f}% of predictions")
print(f"  Within 10%:   {(pct_errors <= 10).sum() / len(pct_errors) * 100:.1f}% of predictions")
print(f"  Within 15%:   {(pct_errors <= 15).sum() / len(pct_errors) * 100:.1f}% of predictions")
print(f"  Within 20%:   {(pct_errors <= 20).sum() / len(pct_errors) * 100:.1f}% of predictions")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nFinal Ridge Model Performance Summary:")
print(f"  Best Alpha:          {best_alpha}")
print(f"  Validation R²:       {val_r2:.4f}")
print(f"  Validation RMSE:     ${val_rmse:,.2f}")
print(f"  Validation MAE:      ${val_mae:,.2f}")
print(f"  Avg Error %:         {pct_errors.mean():.1f}%")
print(f"  Test Predictions:    {len(test_predictions)} houses")
print(f"  Submission file:     data/submission.csv")