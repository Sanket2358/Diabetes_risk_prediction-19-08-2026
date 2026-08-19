from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the logistic regression model
MODEL_PATH = "logistic_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# The exact 17 features the model expects
EXPECTED_FEATURES = [
    'age', 'gender', 'city', 'bmi', 'family_history_diabetes', 
    'physical_activity_level', 'diet_type', 'smoking_status', 
    'alcohol_consumption', 'hours_sleep_per_night', 'stress_level', 
    'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic', 
    'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket'
]

# --- THE COMPLETE HTML INTERFACE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Predictor</title>
    <style>
        /* Dark Theme Variables & Base */
        :root {
            --bg-color: #0f172a;
            --container-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --input-bg: #334155;
            --border: #475569;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: hidden;
        }

        /* Mouse Follower Glow Animation */
        #cursor-glow {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0, 0, 0, 0) 60%);
            border-radius: 50%;
            position: fixed;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 0;
            transition: top 0.1s ease-out, left 0.1s ease-out;
        }

        /* Container Styling */
        .container {
            background-color: var(--container-bg);
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 800px;
            z-index: 1;
            margin: 2rem;
            border: 1px solid var(--border);
        }

        h1 { margin-top: 0; text-align: center; }
        
        /* Grid for the 17 inputs */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .input-group { display: flex; flex-direction: column; }
        .input-group label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            text-transform: capitalize;
        }

        .input-group input {
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.6rem;
            border-radius: 6px;
            outline: none;
            transition: border-color 0.3s;
        }

        .input-group input:focus { border-color: var(--accent); }

        /* Buttons & Results */
        .actions { text-align: center; }
        button {
            background-color: var(--accent);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            font-size: 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.1s;
        }
        button:hover { background-color: var(--accent-hover); transform: translateY(-2px); }
        button:active { transform: translateY(0); }

        #result {
            margin-top: 2rem;
            padding: 1.5rem;
            background-color: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--accent);
            border-radius: 8px;
            display: none;
            text-align: center;
        }
        .prediction-text { font-size: 1.5rem; font-weight: bold; color: var(--accent); margin-bottom: 1rem; }
    </style>
</head>
<body>

    <!-- Mouse Animation Element -->
    <div id="cursor-glow"></div>

    <div class="container">
        <h1>Diabetes Risk Predictor</h1>
        <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">
            Enter numerical values for all features to generate a prediction.
        </p>

        <form id="predictionForm">
            <div class="form-grid">
                <!-- Dynamically generate inputs based on EXPECTED_FEATURES passed from Flask -->
                {% for feature in features %}
                <div class="input-group">
                    <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                    <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" required value="0">
                </div>
                {% endfor %}
            </div>

            <div class="actions">
                <button type="submit">Predict Risk Level</button>
            </div>
        </form>

        <div id="result">
            <div class="prediction-text" id="pred-output"></div>
            <div id="prob-output" style="color: var(--text-secondary);"></div>
        </div>
    </div>

    <script>
        // 1. Mouse Animation Logic
        const cursor = document.getElementById('cursor-glow');
        document.addEventListener('mousemove', (e) => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        });

        // 2. Form Submission & API Call
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const resultDiv = document.getElementById('result');
            const predOutput = document.getElementById('pred-output');
            const probOutput = document.getElementById('prob-output');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const json = await response.json();

                if (response.ok) {
                    resultDiv.style.display = 'block';
                    predOutput.textContent = `Predicted Risk: ${json.prediction}`;
                    
                    // Format probabilities
                    let probString = 'Probabilities: <br>';
                    for (const [key, value] of Object.entries(json.probabilities)) {
                        probString += `${key}: ${(value * 100).toFixed(2)}% | `;
                    }
                    probOutput.innerHTML = probString.slice(0, -2); // remove last pipe
                } else {
                    alert("Error: " + json.error);
                }
            } catch (error) {
                alert("Failed to connect to the server.");
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # Serve the HTML string directly
    return render_template_string(HTML_TEMPLATE, features=EXPECTED_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Ensure columns are in the exact order the model expects
        df = df[EXPECTED_FEATURES]
        
        # Convert all inputs to float (since HTML forms send strings)
        df = df.astype(float)

        prediction = model.predict(df)
        probabilities = model.predict_proba(df).tolist()[0]
        classes = model.classes_.tolist()
        prob_dict = {classes[i]: round(probabilities[i], 4) for i in range(len(classes))}

        return jsonify({
            "prediction": prediction[0],
            "probabilities": prob_dict
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Data type error (ensure inputs are numerical): {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
