import pickle

# 1. Load the pre-trained pipeline artifact
with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# 2. Define the incoming lead data
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# 3. Predict conversion probability
# Scikit-learn pipelines expect a list of dictionaries [lead]
# predict_proba returns an array of shape (n_samples, 2): [P(not converted), P(converted)]
probabilities = pipeline.predict_proba([lead])

# 4. Extract the probability for the positive class (conversion = 1)
conversion_prob = probabilities[0][1]

print(f"Full array: {probabilities}")
print(f"Conversion probability: {conversion_prob:.3f}")