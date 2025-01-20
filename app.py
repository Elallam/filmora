from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from feature_scaling import prepare_features
import pandas as pd



app = Flask(__name__)

# Load the model
model = load('model.joblib')

FEATURES = ['name', 'rating', 'genre', 'year', 'released', 'score', 'votes',
            'director', 'writer', 'star', 'country', 'budget', 'gross', 'company', 'runtime']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features from the request (update these to match your model)
    missing_features = [f for f in FEATURES if f not in data]
    if missing_features:
        return jsonify({
            'error': 'Missing features',
            'missing_features': missing_features
        }), 400
    features = [
        data.get('name', '').lower(),  # Placeholder for handling names
        data.get('rating', '').lower(),  # Rating as categorical
        data.get('genre', '').lower(),  # Genre as categorical
        int(data.get('year', 0)),  # Year as numeric
        data.get('released', '').lower(),  # Placeholder
        float(data.get('score', 0.0)),  # Score as numeric
        int(data.get('votes', 0)),  # Votes as numeric
        data.get('director', '').lower(),  # Director as categorical
        data.get('writer', '').lower(),  # Writer as categorical
        data.get('star', '').lower(),  # Star as categorical
        data.get('country', '').lower(),  # Country as categorical
        float(data.get('budget', 0.0)),  # Budget as numeric
        float(data.get('gross', 0.0)),  # Gross as numeric
        data.get('company', '').lower(),  # Company as categorical
        int(data.get('runtime', 0))  # Runtime as numeric
    ]
    
    # Create a DataFrame for the model
    features_df = pd.DataFrame([features], columns=FEATURES)

    # Convert the features into a 2D array for prediction
    X, y = prepare_features(features_df)
    prediction = model.predict(X)
    print(float(prediction[0]))
    return jsonify({"revenue" : float(math.exp(prediction[0]))})

if __name__ == '__main__':
    app.run(debug=True)
