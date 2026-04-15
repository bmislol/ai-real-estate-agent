import os
import json
from google import genai
from google.genai import types
from pydantic import ValidationError
from dotenv import load_dotenv

load_dotenv()

# Import the schemas
from api.schemas import FeatureExtractionResponse, InterpretationData

# 1. Initialize the Gemini Client
# It automatically looks for an environment variable named GEMINI_API_KEY
client = genai.Client()

# ---------------------------------------------------------
# 2. Define the Two Prompt Variants (Step 8 of the Brief)
# ---------------------------------------------------------

# VARIANT A: The "Strict Rule-Based" Prompt
PROMPT_VARIANT_A = """
You are a highly precise data extraction algorithm for a real estate application.
Your sole job is to extract 10 specific property features from the user's natural language query.

RULES:
1. Map the user's text to the exact categorical values provided in your schema (e.g., if they say "2 story", map it to "2Story").
2. Do NOT hallucinate or guess. If a feature is not explicitly mentioned or clearly implied, leave it as null.
3. For categorical features, use the exact string values defined in the schema.
4. You must evaluate if ALL 10 features were found. If any are null, set is_complete to false.
5. List the exact names of any missing features in the missing_features array.

User Query:
"{user_query}"
"""

# VARIANT B: The "Persona-Driven Expert" Prompt
PROMPT_VARIANT_B = """
You are an expert real estate data analyst. A client has just described a house they want to price. 
I need you to carefully read their description and fill out our standard property intake form.

INSTRUCTIONS:
- Listen carefully to the client's description below.
- Extract the 10 key features required by our pricing model.
- If the client forgot to mention something (like the garage type or year built), do NOT make assumptions. Leave it empty.
- We need to know exactly what the client forgot to mention so we can ask them later. Please set is_complete to false if anything is missing, and list the missing fields.

Client's Description:
"{user_query}"
"""

# ---------------------------------------------------------
# 3. The Extraction Function
# ---------------------------------------------------------
def extract_features(user_query: str, variant: str = "A") -> FeatureExtractionResponse:
    """
    Sends the user's query to Gemini and forces it to return our Pydantic schema.
    """
    # Select the prompt version
    prompt_template = PROMPT_VARIANT_A if variant == "A" else PROMPT_VARIANT_B
    final_prompt = prompt_template.format(user_query=user_query)

    try:
        # Call the Gemini API
        # We use gemini-2.5-flash because it is fast, free, and excellent at JSON extraction.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt,
            config=types.GenerateContentConfig(
                # THIS is the magic! We force Gemini to output strictly to our Pydantic schema.
                response_mime_type="application/json",
                response_schema=FeatureExtractionResponse,
                temperature=0.1 # Low temperature means less creativity, more precision
            ),
        )
        
        # Parse the JSON string response back into our Pydantic object
        extracted_data = FeatureExtractionResponse.model_validate_json(response.text)
        return extracted_data

    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        # The brief requires us to catch validation failures and handle them gracefully.
        raise ValueError("The LLM failed to return valid data matching our schema.")
    except Exception as e:
        print(f"API Error: {e}")
        raise ValueError("An error occurred while communicating with the LLM.")
    
# ---------------------------------------------------------
# 4. Stage 2: Prediction Interpretation (Step 7 of the Brief)
# ---------------------------------------------------------

STAGE_2_PROMPT = """
You are an expert real estate consultant. A machine learning model has just predicted the price of a house based on specific features.
Your job is to interpret this prediction for the client. Do not just restate the number. 
Analyze *why* it costs this much, identify what features are driving the value, and compare it to the broader market.

Baseline Market Statistics (Ames Housing):
- Median Home Price: $163,000
- Typical Price Range: $130,000 to $214,000
- Average Living Area: 1,500 sq ft

House Features Evaluated:
{features_json}

Model's Predicted Price: ${predicted_price}
"""

def interpret_prediction(features_dict: dict, predicted_price: float) -> InterpretationData:
    """
    Feeds the ML prediction and features to Gemini to generate a human-readable analysis.
    """
    # Convert the dictionary of features into a formatted JSON string for the prompt
    features_json = json.dumps(features_dict, indent=2)
    final_prompt = STAGE_2_PROMPT.format(
        features_json=features_json, 
        predicted_price=f"{predicted_price:,.2f}"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InterpretationData, 
                temperature=0.4 # Slightly higher temperature here allows for better narrative flow
            ),
        )
        
        parsed_interpretation = InterpretationData.model_validate_json(response.text)
        return parsed_interpretation

    except ValidationError as e:
        print(f"Stage 2 Validation Error: {e}")
        raise ValueError("Failed to generate a valid interpretation structure.")
    except Exception as e:
        print(f"Stage 2 API Error: {e}")
        raise ValueError("An error occurred while generating the price interpretation.")