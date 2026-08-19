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
            
            /* Status Colors */
            --color-high: #ef4444;      /* Red */
            --color-moderate: #f59e0b; /* Orange */
            --color-low: #10b981;      /* Green */
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
            padding: 2rem 0;
        }

        /* Mouse Follower Glow Animation */
        #cursor-glow {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(0, 0, 0, 0) 60%);
            border-radius: 50%;
            position: fixed;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 0;
            transition: top 0.05s linear, left 0.05s linear;
        }

        /* Container Styling */
        .container {
            background-color: var(--container-bg);
            padding: 2.5rem 3.5rem;
            border-radius: 16px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 900px;
            z-index: 1;
            margin: 2rem;
            border: 1px solid var(--border);
        }

        h1 { margin-top: 0; text-align: center; font-size: 2.2rem; letter-spacing: 0.5px;}
        
        /* Grid for the inputs */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
            margin-bottom: 2.5rem;
        }

        .input-group { display: flex; flex-direction: column; }
        .input-group label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
            text-transform: capitalize;
            font-weight: 500;
        }

        .input-group input, .input-group select {
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.7rem;
            border-radius: 8px;
            outline: none;
            font-size: 1rem;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        .input-group input:focus, .input-group select:focus { 
            border-color: var(--accent); 
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        /* REMOVE ARROWS/SPINNERS FROM NUMBER INPUTS */
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield;
        }

        /* Buttons & Results */
        .actions { 
            display: flex; 
            justify-content: center; 
            gap: 1rem;
        }
        
        button {
            border: none;
            padding: 0.8rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.1s;
        }

        .btn-primary {
            background-color: var(--accent);
            color: white;
        }
        .btn-primary:hover { background-color: var(--accent-hover); transform: translateY(-2px); }
        .btn-primary:active { transform: translateY(0); }

        .btn-secondary {
            background-color: transparent;
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover { background-color: var(--input-bg); transform: translateY(-2px); }
        .btn-secondary:active { transform: translateY(0); }

        /* Dynamic Result Animation & Colors */
        #result {
            margin-top: 2rem;
            padding: 1.5rem;
            background-color: var(--container-bg);
            border: 2px solid var(--accent);
            border-radius: 12px;
            display: none;
            text-align: center;
            opacity: 0;
        }

        @keyframes popIn {
            0% { opacity: 0; transform: scale(0.9) translateY(20px); }
            50% { transform: scale(1.02); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }

        /* Classes added dynamically via JS */
        .result-high { 
            border-color: var(--color-high) !important; 
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
        }
        .result-high .prediction-text { color: var(--color-high); }

        .result-moderate { 
            border-color: var(--color-moderate) !important; 
            box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
        }
        .result-moderate .prediction-text { color: var(--color-moderate); }

        .result-low { 
            border-color: var(--color-low) !important; 
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
        }
        .result-low .prediction-text { color: var(--color-low); }

        .prediction-text { font-size: 1.8rem; font-weight: bold; margin-bottom: 0.8rem; }
    </style>
</head>
<body>

    <!-- Mouse Animation Element -->
    <div id="cursor-glow"></div>

    <div class="container">
        <h1>Diabetes Risk Predictor</h1>
        <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">
            Enter values for all features to generate a prediction.
        </p>

        <form id="predictionForm">
            <div class="form-grid">
                
                <!-- CONTINUOUS / NUMERICAL FEATURES -->
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" step="any" name="age" placeholder="e.g., 45 (Max: 120)" required>
                </div>
                
                <div class="input-group">
                    <label>BMI</label>
                    <input type="number" step="any" name="bmi" placeholder="e.g., 25.5 (Max: 60)" required>
                </div>

                <div class="input-group">
                    <label>Hours Sleep Per Night</label>
                    <input type="number" step="any" name="hours_sleep_per_night" placeholder="e.g., 7 (Max: 24)" required>
                </div>
                
                <div class="input-group">
                    <label>Stress Level</label>
                    <input type="number" step="any" name="stress_level" placeholder="e.g., 4 (Scale 1-10)" required>
                </div>

                <div class="input-group">
                    <label>Fasting Blood Sugar</label>
                    <input type="number" step="any" name="fasting_blood_sugar" placeholder="e.g., 95 (Max: 350)" required>
                </div>

                <div class="input-group">
                    <label>HbA1c Level</label>
                    <input type="number" step="any" name="hba1c_level" placeholder="e.g., 5.5 (Max: 15.0)" required>
                </div>

                <div class="input-group">
                    <label>Blood Pressure Systolic</label>
                    <input type="number" step="any" name="blood_pressure_systolic" placeholder="e.g., 120 (Max: 220)" required>
                </div>

                <div class="input-group">
                    <label>Blood Pressure Diastolic</label>
                    <input type="number" step="any" name="blood_pressure_diastolic" placeholder="e.g., 80 (Max: 130)" required>
                </div>

                <div class="input-group">
                    <label>Waist Circumference (cm)</label>
                    <input type="number" step="any" name="waist_circumference_cm" placeholder="e.g., 90 (Max: 200)" required>
                </div>

                <!-- CATEGORICAL DROPDOWNS -->
                <div class="input-group">
                    <label>Gender</label>
                    <select name="gender" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Family History Diabetes</label>
                    <select name="family_history_diabetes" required>
                        <option value="" disabled selected>Select History</option>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                        <option value="2">Extended</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Smoking Status</label>
                    <select name="smoking_status" required>
                        <option value="" disabled selected>Select Status</option>
                        <option value="0">Non-Smoker</option>
                        <option value="1">Past Smoker</option>
                        <option value="2">Current Smoker</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Alcohol Consumption</label>
                    <select name="alcohol_consumption" required>
                        <option value="" disabled selected>Select Level</option>
                        <option value="0">None</option>
                        <option value="1">Low</option>
                        <option value="2">Moderate</option>
                        <option value="3">High</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label>Physical Activity Level</label>
                    <select name="physical_activity_level" required>
                        <option value="" disabled selected>Select Level</option>
                        <option value="0">Sedentary</option>
                        <option value="1">Low</option>
                        <option value="2">Moderate</option>
                        <option value="3">High</option>
                    </select>
                </div>

                <!-- EXACT OPTIONS FROM DATASET -->
                <div class="input-group">
                    <label>Diet Type</label>
                    <select name="diet_type" required>
                        <option value="" disabled selected>Select Diet</option>
                        <option value="0">Non-Vegetarian</option>
                        <option value="1">Pescatarian</option>
                        <option value="2">Vegan</option>
                        <option value="3">Vegetarian</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>City</label>
                    <select name="city" required>
                        <option value="" disabled selected>Select City</option>
                        <option value="0">Ahmedabad</option>
                        <option value="1">Bengaluru</option>
                        <option value="2">Bhopal</option>
                        <option value="3">Chennai</option>
                        <option value="4">Delhi</option>
                        <option value="5">Hyderabad</option>
                        <option value="6">Indore</option>
                        <option value="7">Jaipur</option>
                        <option value="8">Kanpur</option>
                        <option value="9">Kolkata</option>
                        <option value="10">Lucknow</option>
                        <option value="11">Mumbai</option>
                        <option value="12">Nagpur</option>
                        <option value="13">Patna</option>
                        <option value="14">Pune</option>
                        <option value="15">Surat</option>
                        <option value="16">Thane</option>
                        <option value="17">Visakhapatnam</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Income Bracket</label>
                    <select name="income_bracket" required>
                        <option value="" disabled selected>Select Bracket</option>
                        <option value="0">High</option>
                        <option value="1">Low</option>
                        <option value="2">Middle</option>
                    </select>
                </div>
            </div>

            <div class="actions">
                <button type="submit" class="btn-primary">Predict Risk Level</button>
                <button type="button" class="btn-secondary" id="predictMoreBtn">Predict More</button>
            </div>
        </form>

        <div id="result">
            <div class="prediction-text" id="pred-output"></div>
            <div id="prob-output" style="color: var(--text-secondary);"></div>
        </div>
    </div>

    <script>
        // Mouse Animation Logic
        const cursor = document.getElementById('cursor-glow');
        document.addEventListener('mousemove', (e) => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        });

        // Predict More (Reset) Logic
        document.getElementById('predictMoreBtn').addEventListener('click', () => {
            document.getElementById('predictionForm').reset();
            const resultDiv = document.getElementById('result');
            resultDiv.style.animation = 'none';
            resultDiv.style.display = 'none';
        });

        // Form Submission & API Call
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const resultDiv = document.getElementById('result');
            const predOutput = document.getElementById('pred-output');
            const probOutput = document.getElementById('prob-output');

            // Change button text while loading
            const submitBtn = e.target.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Processing...';
            submitBtn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const json = await response.json();

                if (response.ok) {
                    // Update Text
                    predOutput.textContent = `Predicted Risk: ${json.prediction}`;
                    
                    let probString = 'Probabilities: <br>';
                    for (const [key, value] of Object.entries(json.probabilities)) {
                        probString += `${key}: ${(value * 100).toFixed(2)}% &nbsp;|&nbsp; `;
                    }
                    probOutput.innerHTML = probString.slice(0, -14); // Remove last separator
                    
                    // Reset classes for colors
                    resultDiv.className = ''; 
                    
                    // Apply dynamic color class based on prediction
                    if (json.prediction === 'High') {
                        resultDiv.classList.add('result-high');
                    } else if (json.prediction === 'Moderate') {
                        resultDiv.classList.add('result-moderate');
                    } else if (json.prediction === 'Low') {
                        resultDiv.classList.add('result-low');
                    }

                    // Trigger Success Animation (Pop-in)
                    resultDiv.style.display = 'block';
                    resultDiv.style.animation = 'none';
                    resultDiv.offsetHeight; /* trigger reflow to restart animation */
                    resultDiv.style.animation = 'popIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards';
                    
                    // Scroll to results
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    alert("Error: " + json.error);
                }
            } catch (error) {
                alert("Failed to connect to the server.");
            } finally {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        df = df[EXPECTED_FEATURES]
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
