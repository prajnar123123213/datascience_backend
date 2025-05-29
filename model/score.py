from flask import Blueprint, request, jsonify
import pandas as pd
import os
from sklearn.preprocessing import QuantileTransformer
import numpy as np

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'synthetic_data_science_scores.csv')
data = pd.read_csv(data_path)

# Dynamically set n_quantiles based on number of samples
n_samples = data.shape[0]
n_quantiles = min(1000, n_samples)

# Fit quantile transformers
from sklearn.preprocessing import QuantileTransformer
import numpy as np

mcq_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
frq_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')

mcq_transformer.fit(data[['mcq']])
frq_transformer.fit(data[['frq']])


# Score class definition
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

# Define the blueprint
score_api = Blueprint('score_api', __name__)

@score_api.route('/api/percentile', methods=['POST'])
def calculate_percentile():
    scores = request.get_json()
    mcq_score = scores.get('mcq')
    frq_score = scores.get('frq')

    if mcq_score is None or frq_score is None:
        return jsonify({'error': 'Missing mcq or frq scores'}), 400

    score = Score(mcq_score, frq_score)
    return jsonify({
        'mcq_percentile': score.mcq_percentile(),
        'frq_percentile': score.frq_percentile()
    })
