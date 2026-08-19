from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the logistic regression model
MODEL_PATH = "logistic_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# The exact 17 features the model expects based on your pickle file
EXPECTED_FEATURES = [
    'age', 'gender', 'city', 'bmi', 'family_history_diabetes', 
    'physical_activity_level', 'diet_type', 'smoking_status', 
    'alcohol_consumption', 'hours_sleep_per_night', 'stress_level', 
    'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic', 
    'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Logistic Regression Model API is running. Send a POST request to /predict."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Create a DataFrame from the JSON data
        df = pd.DataFrame([data])
        
        # Ensure columns are in the exact order the model expects
        df = df[EXPECTED_FEATURES]

        # Make prediction
        prediction = model.predict(df)
        
        # Get prediction probabilities for each class (High, Low, Moderate)
        probabilities = model.predict_proba(df).tolist()[0]
        classes = model.classes_.tolist()
        prob_dict = {classes[i]: round(probabilities[i], 4) for i in range(len(classes))}

        return jsonify({
            "prediction": prediction[0],
            "probabilities": prob_dict
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render assigns a port dynamically via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
