from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
import uuid

# ---------------------------------------------------------
# 1. Core Feature Schema (LLM targets these fields)
# ---------------------------------------------------------
class PropertyFeatures(BaseModel):
    """The 10 key features required by our Random Forest model."""
    
    # Nominal (Categorical)
    neighborhood: Optional[str] = Field(None, description="The neighborhood name (e.g., CollgCr, Edwards, NoRidge).")
    house_style: Optional[str] = Field(None, description="Style of dwelling (e.g., 1Story, 2Story, SplitFoyer).")
    garage_type: Optional[str] = Field(None, description="Location of garage (e.g., Attchd, Detchd, None).")
    
    # Ordinal (Ranked Categorical)
    exter_qual: Optional[Literal['Ex', 'Gd', 'TA', 'Fa', 'Po']] = Field(None, description="Evaluates the quality of the material on the exterior (Excellent, Good, Typical/Average, Fair, Poor).")
    bsmt_qual: Optional[Literal['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']] = Field(None, description="Evaluates the height of the basement. Use 'None' if no basement.")
    
    # Continuous / Discrete (Numeric)
    overall_qual: Optional[int] = Field(None, ge=1, le=10, description="Rates the overall material and finish of the house (1-10 scale).")
    gr_liv_area: Optional[int] = Field(None, description="Above grade (ground) living area square feet.")
    lot_frontage: Optional[int] = Field(None, description="Linear feet of street connected to property.")
    year_built: Optional[int] = Field(None, description="Original construction date.")
    full_bath: Optional[int] = Field(None, ge=0, description="Full bathrooms above grade.")

# ---------------------------------------------------------
# 2. Stage 1 LLM Extraction Schema (The Completeness Signal)
# ---------------------------------------------------------
class FeatureExtractionResponse(BaseModel):
    """The structured output we demand from the LLM in Stage 1."""
    
    extracted_features: PropertyFeatures = Field(
        description="The features successfully extracted from the user's natural language query."
    )
    is_complete: bool = Field(
        description="True ONLY if all 10 features were confidently extracted. False if any are missing."
    )
    missing_features: List[str] = Field(
        description="A list of the exact feature names that were NOT found in the user's query."
    )

class InterpretationData(BaseModel):
    """Structured breakdown of the AI's price interpretation."""
    price_context: str = Field(
        description="Explanation of whether the predicted price is high, low, or average compared to the baseline market."
    )
    key_driving_factors: List[str] = Field(
        description="Top 3 features from the user's query that most heavily influenced this specific price."
    )
    market_comparison: str = Field(
        description="A brief narrative comparing this property against the provided median market statistics."
    )

class CombinedResponse(BaseModel):
    """The final payload sent back to the UI (Step 9 Requirement)."""
    features_used: PropertyFeatures = Field(description="The exact features fed into the ML model.")
    predicted_price: float = Field(description="The raw dollar output from the Random Forest model.")
    interpretation: InterpretationData = Field(description="The LLM's Stage 2 analysis of the prediction.")