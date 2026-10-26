import pickle

# 1. Define the lead data exactly as provided in the question
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# 2. Load the pipeline model from the file
model_file = './05-deployment/pipeline.bin'
with open(model_file, 'rb') as f_in:
    artifact = pickle.load(f_in)

# 3. Check if the artifact is a Pipeline object or a tuple, then make a prediction
    # Fallback: if it was saved as a tuple (dv, model)
dv, model = artifact
X = dv.transform(lead)
probabilities = model.predict_proba(X)[0]

# 4. Print the final probability of conversion
# The target probability is for conversion (typically class index 1)
print(f"Full probabilities array: {probabilities}")
print(f"Probability of conversion: {probabilities[1]:.3f}")
