from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    
    # Extract input features (assuming these are the features the model expects)
    age = float(data['age'])
    bmi = float(data['bmi'])
    blood_pressure = float(data['blood_pressure'])
    
    # Create a feature array (you can add more features as needed)
    input_features = np.array([[age, bmi, blood_pressure]])
    
    # Predict using the loaded model
    prediction = model.predict(input_features)
    
    # Return the prediction as a JSON response
    result = "Diabetic" if prediction[0] == 1 else "Non-diabetic"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
