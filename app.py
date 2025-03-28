from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([
        data['ChemicalName'],
        data['CompanyName'],
        data['BrandName']
    ]).reshape(1, -1)
    prediction = model.predict(input_data)
    predicted_category = le.inverse_transform(prediction)[0]
    return jsonify({'PrimaryCategory': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)