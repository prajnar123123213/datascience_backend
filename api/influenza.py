from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Setup API blueprint
influenza_api = Blueprint('influenza_api', __name__, url_prefix='/api/influenza')
api = Api(influenza_api)

class InfluenzaModel:
    def __init__(self):
        # Use absolute path to avoid file not found errors
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'influenza_survival.csv')
        df = pd.read_csv(file_path)

        self.le = LabelEncoder()
        df['sex'] = self.le.fit_transform(df['sex'])

        self.features = ['age', 'sex', 'fever', 'cough', 'shortness_of_breath', 'chest_pain', 'comorbidities']
        X = df[self.features]
        y = df['survived']

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict(self, data):
        df = pd.DataFrame([data])
        df['sex'] = self.le.transform(df['sex'])
        probs = self.model.predict_proba(df[self.features])[0]
        return {
            'died': round(float(probs[0]), 2),
            'survived': round(float(probs[1]), 2)
        }

# Create model instance once
model_instance = InfluenzaModel()

# API Resource
class SurviveOrThrive(Resource):
    def post(self):
        data = request.get_json()
        prediction = model_instance.predict(data)
        return jsonify(prediction)

# Register route
api.add_resource(SurviveOrThrive, '/survive-or-thrive')
