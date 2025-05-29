# api/score.py

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
import os
from model.score import Score
from sklearn.preprocessing import QuantileTransformer

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), 'synthetic_data_science_scores.csv')
data = pd.read_csv(data_path)

# Prepare Quantile Transformers
n_samples = data.shape[0]
n_quantiles = min(1000, n_samples)

mcq_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
frq_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')

mcq_transformer.fit(data[['mcq']])
frq_transformer.fit(data[['frq']])

# Score logic class
class Score:
    def __init__(self, mcq, frq):
        self.mcq = mcq
        self.frq = frq

    def mcq_percentile(self):
        percentile = mcq_transformer.transform([[self.mcq]])[0][0]
        return float(np.round(percentile * 100, 2))

    def frq_percentile(self):
        percentile = frq_transformer.transform([[self.frq]])[0][0]
        return float(np.round(percentile * 100, 2))

# Set up API blueprint and restful API
score_api = Blueprint('score_api', __name__, url_prefix='/api/score')
api = Api(score_api)

# Define REST resource
class ScoreAPI(Resource):
    def post(self):
        data = request.get_json()
        mcq_score = data.get('mcq')
        frq_score = data.get('frq')

        if mcq_score is None or frq_score is None:
            return {'error': 'Missing mcq or frq scores'}, 400

        score = Score(mcq_score, frq_score)
        return {
            'mcq_percentile': score.mcq_percentile(),
            'frq_percentile': score.frq_percentile()
        }

# Add resource route
api.add_resource(ScoreAPI, '/percentile')
