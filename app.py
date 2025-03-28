from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model safely
try:
    with open("model.pkl", "rb") as f:  
        model = pickle.load(f)
        print("✅ Model loaded successfully.")
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    print("⚠️ Error: model.pkl is missing or corrupted. Using dummy prediction.")
    model = lambda x: ["Dummy Prediction"]  # Fallback function

# Home route - Serve HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON input from frontend
        input_df = pd.DataFrame([data])  # Convert input to DataFrame
        prediction = model.predict(input_df)  # Make prediction
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
