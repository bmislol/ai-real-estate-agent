import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.llm import extract_features

# Load the API key
load_dotenv()

print("--- Starting Prompt Versioning Test (Step 8) ---\n")

# 1. Define our 3 Test Queries
# We want to test different user behaviors: a perfect user, a vague user, and a messy user.
test_queries = [
    # Test 1: Vague/Incomplete (Should trigger is_complete = False)
    "How much is a house in Edwards with a detached garage?",
    
    # Test 2: Perfect & Complete
    "I have a 1Story house in CollgCr. It has an Attchd garage, Excellent exterior quality, Good basement, overall quality of 7, 1500 sqft living area, 60 ft lot frontage, built in 2005, with 2 full baths.",
    
    # Test 3: Messy/Colloquial (Tests the LLM's ability to map weird text to our strict categories)
    "looking for a place in NoRidge. two stories, no basement at all, average exterior. 3 bathrooms. 2500 sq ft. built in 1990. no garage. lot is 80ft. overall finish is an 8 out of 10."
]

# 2. Run the tests
for i, query in enumerate(test_queries, 1):
    print(f"==================================================")
    print(f"TEST QUERY {i}: \"{query}\"")
    print(f"==================================================")
    
    for variant in ["A", "B"]:
        print(f"\n---> Running VARIANT {variant}...")
        try:
            # Call our Gemini function!
            result = extract_features(query, variant=variant)
            
            # Print the structured output
            print(f"Is Complete?      {result.is_complete}")
            if not result.is_complete:
                print(f"Missing Features: {result.missing_features}")
            print(f"Extracted Data:   {result.extracted_features.model_dump_json(indent=2)}")
            
        except Exception as e:
            print(f"Variant {variant} FAILED: {e}")
            
    print("\n")