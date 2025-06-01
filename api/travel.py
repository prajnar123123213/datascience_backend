from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os

# Setup API blueprint
travel_api = Blueprint('travel_api', __name__, url_prefix='/api/travel')
api = Api(travel_api)

class TravelModel:
    def __init__(self):
        # Load dataset using absolute path
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'travel_recommendation.csv')
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ['destination', 'season', 'activity', 'budget', 'continent', ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file.")

        # Encode categorical variables
        self.encoders = {}
        columns_to_encode = ['destination', 'season', 'activity', 'budget', 'continent']
        for col in columns_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le

        self.destinations = df['destination']
        X = df[['season', 'activity', 'budget', 'continent']]
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(X, self.destinations)

    def predict(self, data):
        encoded = []
        for feature in ['season', 'activity', 'budget','continent']:
            val = self.encoders[feature].transform([data[feature]])[0]
            encoded.append(val)
        destination = self.model.predict([encoded])[0]
        return {'recommended_destination': destination}

# Instantiate model once
model_instance = TravelModel()

# API Resource
class RecommendDestination(Resource):
    def post(self):
        data = request.get_json()
        prediction = model_instance.predict(data)
        return jsonify(prediction)

# Register endpoint
api.add_resource(RecommendDestination, '/recommend')
