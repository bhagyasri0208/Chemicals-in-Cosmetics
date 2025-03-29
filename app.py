from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define category mapping (Ensure this matches model training categories)
category_mapping = {
    "Cosmetics": "Cosmetics",
    "Pharmaceuticals": "Pharmaceuticals",
    "Food Additives": "Food Additives",
    "Industrial Chemicals": "Industrial Chemicals",
    "Household Products": "Household Products",
    "Agricultural Chemicals": "Agricultural Chemicals"
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure input has correct columns
        required_columns = ['ChemicalName', 'CompanyName', 'BrandName']
        if not all(col in input_data.columns for col in required_columns):
            return jsonify({'error': 'Missing required input fields'}), 400
        
        # Combine text columns and apply vectorizer transformation
        input_text = input_data[required_columns].apply(lambda x: ' '.join(x), axis=1)
        input_vectorized = vectorizer.transform(input_text)
        
        # Make prediction
        prediction = model.predict(input_vectorized)[0]
        print(f"Raw Prediction Output: {prediction}")  # Debugging
        
        # Convert prediction to category name
        category_name = category_mapping.get(prediction, f" Category ({prediction})")
        
        # Return prediction result
        return jsonify({'PrimaryCategory': category_name})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
