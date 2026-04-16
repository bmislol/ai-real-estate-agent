import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.llm import extract_features, interpret_prediction
from api.schemas import FeatureExtractionResponse, CombinedResponse

# 1. Global variable to hold our machine learning model
ml_model = None

# 2. FastAPI Lifespan (Loads the model once when the server starts)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    try:
        print("Loading Machine Learning Model...")
        ml_model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
    yield
    # Cleanup (if needed) when server shuts down
    ml_model = None

# 3. Initialize FastAPI
app = FastAPI(
    title="AI Real Estate Agent API",
    description="Two-stage LLM Prompt Chain with ML Prediction",
    lifespan=lifespan
)

# 4. Input Schema for the API route
class UserQuery(BaseModel):
    text: str

# 5. The Main POST Route
@app.post("/predict", response_model=None) # response_model=None because we return different schemas based on completeness
async def predict_property_price(query: UserQuery):
    try:
        # ==========================================
        # STAGE 1: Extract Features from Text
        # ==========================================
        print(f"Processing query: {query.text}")
        extraction: FeatureExtractionResponse = extract_features(query.text)
        
        # --- QoL FIX: Validate Neighborhoods ---
        VALID_NEIGHBORHOODS = [
            "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", 
            "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", 
            "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", 
            "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"
        ]
        
        # If the LLM found a neighborhood, but it's not in our training data list...
        if extraction.extracted_features.neighborhood and extraction.extracted_features.neighborhood not in VALID_NEIGHBORHOODS:
            extraction.extracted_features.neighborhood = None # Erase the bad guess
            if "neighborhood" not in extraction.missing_features:
                extraction.missing_features.append("neighborhood")
            extraction.is_complete = False
        # ---------------------------------------

        # If the user forgot information, stop the chain and ask them for it!
        if not extraction.is_complete:
            print("Extraction incomplete. Asking user for missing features.")
            return extraction
        
        # ==========================================
        # STAGE 1.5: Machine Learning Prediction
        # ==========================================
        print("Extraction complete. Running ML Prediction...")
        
        # Convert the Pydantic model into a dictionary, then into a 1-row Pandas DataFrame
        features_dict = extraction.extracted_features.model_dump()
        input_df = pd.DataFrame([features_dict])
        
        # --- THE FIX: Rename snake_case columns to PascalCase for Scikit-Learn ---
        column_mapping = {
            "neighborhood": "Neighborhood",
            "house_style": "HouseStyle",
            "garage_type": "GarageType",
            "exter_qual": "ExterQual",
            "bsmt_qual": "BsmtQual",
            "overall_qual": "OverallQual",
            "gr_liv_area": "GrLivArea",
            "lot_frontage": "LotFrontage",
            "year_built": "YearBuilt",
            "full_bath": "FullBath"
        }
        input_df = input_df.rename(columns=column_mapping)
        # -----------------------------------------------------------------------
        
        # Predict the price using our saved Random Forest model
        prediction_array = ml_model.predict(input_df)
        predicted_price = float(prediction_array[0])
        
        # ==========================================
        # STAGE 2: Price Interpretation
        # ==========================================
        print(f"Model Predicted: ${predicted_price:,.2f}. Generating interpretation...")
        interpretation = interpret_prediction(features_dict, predicted_price)
        
        # ==========================================
        # FINAL: Build the Combined Response
        # ==========================================
        final_response = CombinedResponse(
            features_used=extraction.extracted_features,
            predicted_price=predicted_price,
            interpretation=interpretation
        )
        
        return final_response

    except ValueError as ve:
        # Catch our custom LLM validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch unexpected server/ML errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 6. Health Check Route (Good practice for Docker)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": ml_model is not None}