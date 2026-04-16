import os
import streamlit as st
import requests
from dotenv import load_dotenv

# The URL where your FastAPI server is running
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="centered")

# ==========================================
# 1. State Management (The App's Memory)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "🏠 **Welcome!** Describe the property you are looking to price in plain English.\n\n*Supported Neighborhoods:* `Edwards`, `CollgCr`, `NoRidge`, `NridgHt`, `OldTown`, `Somerst`, `Mitchel`, `NAmes`, `NWAmes`, `Sawyer`, `Timber`, `Gilbert`, `ClearCr`, `Crawfor`, `BrkSide`, `IDOTRR`, `MeadowV`, `SWISU`, `StoneBr`, `Veenker`, `Blmngtn`, `BrDale`, `NPkVill`, `Blueste`.\n\n*(e.g., 'I have a 1-story house in Edwards with 2 baths')*"
        }
    ]
if "awaiting_form" not in st.session_state:
    st.session_state.awaiting_form = False
if "missing_features" not in st.session_state:
    st.session_state.missing_features = []
if "original_query" not in st.session_state:
    st.session_state.original_query = ""

# ==========================================
# 2. UI Layout: The Chat Interface
# ==========================================
st.title("🏠 AI Real Estate Agent")
st.markdown("---")

# Display the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 3. UI Layout: The Fallback Form
# ==========================================
if st.session_state.awaiting_form:
    st.info("I found some of the details, but I need a little more information to run the pricing model:")
    
    with st.form("missing_data_form"):
        form_answers = {}
        
        # Human-Readable Form Generation
        for feature in st.session_state.missing_features:
            
            # 1. Numerical Inputs
            if feature == "overall_qual":
                form_answers[feature] = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
            elif feature == "full_bath":
                form_answers[feature] = st.number_input("Number of Full Bathrooms", min_value=0, step=1)
            elif feature == "gr_liv_area":
                form_answers[feature] = st.number_input("Above Ground Living Area (Square Feet)", min_value=100, step=10)
            elif feature == "lot_frontage":
                form_answers[feature] = st.number_input("Lot Frontage (Linear Feet)", min_value=0, step=5)
            elif feature == "year_built":
                form_answers[feature] = st.number_input("Year Built", min_value=1800, max_value=2026, step=1)
                
            # 2. Categorical Dropdowns (Human Readable!)
            elif feature == "exter_qual":
                form_answers[feature] = st.selectbox("Exterior Quality", ["Excellent", "Good", "Typical/Average", "Fair", "Poor"])
            elif feature == "bsmt_qual":
                form_answers[feature] = st.selectbox("Basement Quality", ["Excellent", "Good", "Typical/Average", "Fair", "Poor", "None (No Basement)"])
            elif feature == "garage_type":
                form_answers[feature] = st.selectbox("Garage Type", ["Attached", "Detached", "Built-In", "Basement", "Car Port", "2 Types", "None (No Garage)"])
            elif feature == "house_style":
                form_answers[feature] = st.selectbox("House Style", ["1 Story", "2 Story", "1.5 Story Finished", "1.5 Story Unfinished", "Split Foyer", "Split Level", "2.5 Story Unfinished", "2.5 Story Finished"])
            elif feature == "neighborhood":
                form_answers[feature] = st.selectbox("Neighborhood", ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"])
            else:
                form_answers[feature] = st.text_input(f"{feature.replace('_', ' ').title()}")
                
        submit_button = st.form_submit_button("Submit Details")
        
        if submit_button:
            # We stitch the human-readable answers together for the LLM
            clarification_text = ". ".join([f"The {k.replace('_', ' ')} is {v}" for k, v in form_answers.items()])
            combined_query = st.session_state.original_query + ". " + clarification_text
            
            st.session_state.messages.append({"role": "user", "content": "*(Submitted missing form data)*"})
            
            with st.spinner("Analyzing complete data and predicting price..."):
                response = requests.post(API_URL, json={"text": combined_query})
                
            if response.status_code == 200:
                data = response.json()
                
                # THE FIX: Check if the LLM STILL thinks we are missing data!
                if "missing_features" in data:
                    st.warning("Hmm, some of that data still wasn't quite right. Let's try again.")
                    st.session_state.missing_features = data["missing_features"]
                    st.rerun() # Refresh the form to try again
                
                # If we finally have a perfect prediction, show it!
                else:
                    price = data.get("predicted_price", 0)
                    interpretation = data.get("interpretation", {})
                    
                    final_msg = f"### Predicted Price: **${price:,.2f}**\n\n"
                    final_msg += f"**Analysis:** {interpretation.get('price_context', '')}\n\n"
                    final_msg += f"**Key Drivers:** {', '.join(interpretation.get('key_driving_factors', []))}\n\n"
                    final_msg += f"**Market Comparison:** {interpretation.get('market_comparison', '')}"
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_msg})
                    st.session_state.awaiting_form = False # Close the form
                    st.rerun()
            else:
                # Try to extract the exact error message from FastAPI
                try:
                    error_msg = response.json().get("detail", response.text)
                except:
                    error_msg = response.text
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ API Error: {error_msg}"})
                st.session_state.awaiting_form = False
                st.rerun()

# ==========================================
# 4. Initial User Input (The Chat Bar)
# ==========================================
# Only show the chat bar if we are NOT waiting for them to fill out a form
elif prompt := st.chat_input("E.g., I want a 1-story house in CollgCr..."):
    # Save what they typed
    st.session_state.original_query = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display it immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.spinner("Extracting features..."):
        # Hit your FastAPI server
        response = requests.post(API_URL, json={"text": prompt})
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if FastAPI hit the "is_complete: False" block
            if "missing_features" in data:
                st.session_state.missing_features = data["missing_features"]
                st.session_state.awaiting_form = True
                st.rerun() # Refresh the page to show the form
            
            # Otherwise, FastAPI gave us a perfect price immediately!
            else:
                price = data.get("predicted_price", 0)
                interpretation = data.get("interpretation", {})
                
                final_msg = f"### Predicted Price: **${price:,.2f}**\n\n"
                final_msg += f"**Analysis:** {interpretation.get('price_context', '')}\n\n"
                final_msg += f"**Key Drivers:** {', '.join(interpretation.get('key_driving_factors', []))}\n\n"
                final_msg += f"**Market Comparison:** {interpretation.get('market_comparison', '')}"
                
                st.session_state.messages.append({"role": "assistant", "content": final_msg})
                st.rerun()
        else:
            try:
                error_msg = response.json().get("detail", response.text)
            except:
                error_msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ API Error: {error_msg}"})
            st.rerun()