import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

print("--- Starting Model Training Process ---")

# 1. Setup Directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# 2. Load the RAW splits (The pipeline needs to learn the cleaning steps from scratch)
train_df = pd.read_csv(os.path.join(data_dir, 'train_raw.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'val_raw.csv'))

# Separate features (X) and target (y)
X_train = train_df.drop(columns=['SalePrice'])
y_train = train_df['SalePrice']

X_val = val_df.drop(columns=['SalePrice'])
y_val = val_df['SalePrice']

# 3. Define our Feature Groups (Exactly as we did in EDA)
num_cols = ['GrLivArea', 'LotFrontage', 'YearBuilt', 'FullBath', 'OverallQual']
ord_cols = ['ExterQual', 'BsmtQual']
nom_cols = ['Neighborhood', 'HouseStyle', 'GarageType']

# 4. Build the Preprocessing Pipelines
# A. Numeric Pipeline: Fill missing with median -> Scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# B. Ordinal Pipeline: Fill missing with 'None' -> Ordinal Encode -> Scale
exter_cats = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
bsmt_cats = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('ordinal', OrdinalEncoder(categories=[exter_cats, bsmt_cats], handle_unknown='use_encoded_value', unknown_value=-1)),
    ('scaler', StandardScaler())
])

# C. Nominal Pipeline: Fill missing with 'None' -> One-Hot Encode (No scaling needed for 0s and 1s)
nom_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 5. Combine everything into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('numeric', num_pipeline, num_cols),
    ('ordinal', ord_pipeline, ord_cols),
    ('nominal', nom_pipeline, nom_cols)
])

# 6. Create the Final Full Pipelines (Preprocessor + Model)
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_estimators=100))
])

# 7. Train Both Models
print("Training Linear Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)

# 8. Evaluate Models on Validation Set
# We use Root Mean Squared Error (RMSE) because the output is in real dollars
lr_preds = lr_pipeline.predict(X_val)
rf_preds = rf_pipeline.predict(X_val)

lr_rmse = root_mean_squared_error(y_val, lr_preds)
rf_rmse = root_mean_squared_error(y_val, rf_preds)

print(f"\n--- Validation Results ---")
print(f"Linear Regression RMSE: ${lr_rmse:,.2f}")
print(f"Random Forest RMSE:     ${rf_rmse:,.2f}")

# 9. Determine Winner and Save
if rf_rmse < lr_rmse:
    best_model = rf_pipeline
    winner = "Random Forest"
else:
    best_model = lr_pipeline
    winner = "Linear Regression"

print(f"\nWinning Model: {winner}! Saving to /models/best_model.pkl...")

model_path = os.path.join(models_dir, 'best_model.pkl')
joblib.dump(best_model, model_path)

print("Process Complete. Model saved and ready for FastAPI.")