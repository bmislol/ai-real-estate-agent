import os
import streamlit as st
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# 0. Setup & Database Connection
# ==========================================
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="wide") # Changed to wide layout for the sidebar

# Initialize Supabase client (Cached so it doesn't reconnect on every button click)
@st.cache_resource
def init_db() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_db()

# ==========================================
# 1. State Management & Helper Functions
# ==========================================
WELCOME_MSG = "🏠 **Welcome!** Describe the property you are looking to price in plain English.\n\n*Supported Neighborhoods:* `Edwards`, `CollgCr`, `NoRidge`, `NridgHt`, `OldTown`, `Somerst`, `Mitchel`, `NAmes`, `NWAmes`, `Sawyer`, `Timber`, `Gilbert`, `ClearCr`, `Crawfor`, `BrkSide`, `IDOTRR`, `MeadowV`, `SWISU`, `StoneBr`, `Veenker`, `Blmngtn`, `BrDale`, `NPkVill`, `Blueste`.\n\n*(e.g., 'I have a 1-story house in Edwards with 2 baths')*"

def create_new_chat():
    """Creates a new chat in the DB and resets the UI state."""
    # Insert a new folder into the 'chats' table
    response = supabase.table("chats").insert({"title": "New Property Valuation"}).execute()
    new_chat_id = response.data[0]['id']
    
    # Save the welcome message to the 'messages' table
    supabase.table("messages").insert({"chat_id": new_chat_id, "role": "assistant", "content": WELCOME_MSG}).execute()
    
    # Update Streamlit's memory
    st.session_state.current_chat_id = new_chat_id
    st.session_state.awaiting_form = False
    st.session_state.missing_features = []
    st.session_state.original_query = ""

def save_message(role: str, content: str):
    """Saves a single message to the DB."""
    supabase.table("messages").insert({
        "chat_id": st.session_state.current_chat_id, 
        "role": role, 
        "content": content
    }).execute()

# Initialize the very first chat if they just opened the app
if "current_chat_id" not in st.session_state:
    # Check if they have ANY past chats
    existing_chats = supabase.table("chats").select("id").order("created_at", desc=True).limit(1).execute()
    if existing_chats.data:
        st.session_state.current_chat_id = existing_chats.data[0]['id']
    else:
        create_new_chat()

if "awaiting_form" not in st.session_state:
    st.session_state.awaiting_form = False
if "missing_features" not in st.session_state:
    st.session_state.missing_features = []
if "original_query" not in st.session_state:
    st.session_state.original_query = ""

# ==========================================
# 2. Sidebar Layout (Chat History)
# ==========================================
with st.sidebar:
    st.title("Conversations")
    if st.button("➕ New Valuation", use_container_width=True):
        create_new_chat()
        st.rerun()
        
    st.divider()
    
    # Fetch all past chats from the DB to show in the sidebar
    chats_res = supabase.table("chats").select("id, title").order("created_at", desc=True).execute()
    for chat in chats_res.data:
        # If they click an old chat, load it!
        if st.button(f"🏠 {chat['title']}", key=chat['id'], use_container_width=True):
            st.session_state.current_chat_id = chat['id']
            st.session_state.awaiting_form = False # Reset form state when switching chats
            st.rerun()

# ==========================================
# 3. Main Chat Interface
# ==========================================
st.title("🏠 AI Real Estate Agent")
st.markdown("---")

# Fetch and display all messages for the CURRENTLY selected chat
messages_res = supabase.table("messages").select("role, content").eq("chat_id", st.session_state.current_chat_id).order("created_at").execute()
for msg in messages_res.data:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. Fallback Form Logic
# ==========================================
if st.session_state.awaiting_form:
    st.info("I found some of the details, but I need a little more information to run the pricing model:")
    
    with st.form("missing_data_form"):
        form_answers = {}
        for feature in st.session_state.missing_features:
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
            clarification_text = ". ".join([f"The {k.replace('_', ' ')} is {v}" for k, v in form_answers.items()])
            combined_query = st.session_state.original_query + ". " + clarification_text
            
            # SAVE to DB instead of just Streamlit memory
            save_message("user", "*(Submitted missing form data)*")
            
            with st.spinner("Analyzing complete data and predicting price..."):
                response = requests.post(API_URL, json={"text": combined_query})
            
            if response.status_code == 200:
                data = response.json()
                if "missing_features" in data:
                    st.warning("Hmm, some of that data still wasn't quite right. Let's try again.")
                    st.session_state.missing_features = data["missing_features"]
                    st.rerun() 
                else:
                    price = data.get("predicted_price", 0)
                    interpretation = data.get("interpretation", {})
                    
                    final_msg = f"### Predicted Price: **${price:,.2f}**\n\n"
                    final_msg += f"**Analysis:** {interpretation.get('price_context', '')}\n\n"
                    final_msg += f"**Key Drivers:** {', '.join(interpretation.get('key_driving_factors', []))}\n\n"
                    final_msg += f"**Market Comparison:** {interpretation.get('market_comparison', '')}"
                    
                    # SAVE to DB
                    save_message("assistant", final_msg)
                    st.session_state.awaiting_form = False 
                    st.rerun()
            else:
                try:
                    error_msg = response.json().get("detail", response.text)
                except:
                    error_msg = response.text
                
                # SAVE error to DB
                save_message("assistant", f"⚠️ API Error: {error_msg}")
                st.session_state.awaiting_form = False
                st.rerun()

# ==========================================
# 5. Initial User Input (The Chat Bar)
# ==========================================
# Check if the LAST message in the DB was from the user. If so, we need to process it!
if messages_res.data and messages_res.data[-1]["role"] == "user" and not st.session_state.awaiting_form and messages_res.data[-1]["content"] != "*(Submitted missing form data)*":
    latest_user_prompt = messages_res.data[-1]["content"]
    
    with st.spinner("Extracting features..."):
        response = requests.post(API_URL, json={"text": latest_user_prompt})
        
        if response.status_code == 200:
            data = response.json()
            if "missing_features" in data:
                st.session_state.missing_features = data["missing_features"]
                st.session_state.awaiting_form = True
                st.rerun() 
            else:
                price = data.get("predicted_price", 0)
                interpretation = data.get("interpretation", {})
                
                final_msg = f"### Predicted Price: **${price:,.2f}**\n\n"
                final_msg += f"**Analysis:** {interpretation.get('price_context', '')}\n\n"
                final_msg += f"**Key Drivers:** {', '.join(interpretation.get('key_driving_factors', []))}\n\n"
                final_msg += f"**Market Comparison:** {interpretation.get('market_comparison', '')}"
                
                save_message("assistant", final_msg)
                st.rerun()
        else:
            try:
                error_msg = response.json().get("detail", response.text)
            except:
                error_msg = response.text
            save_message("assistant", f"⚠️ API Error: {error_msg}")
            st.rerun()

elif prompt := st.chat_input("E.g., I want a 1-story house in CollgCr..."):
    st.session_state.original_query = prompt
    save_message("user", prompt)
    st.rerun()