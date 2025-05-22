# api/score.py

from flask import Blueprint, request, jsonify
import pandas as pd
from scipy.stats import percentileofscore
import os

# Define the blueprint
score_api = Blueprint('score_api', __name__)

# Load the dataset (located in the same folder as this file)
data_path = os.path.join(os.path.dirname(__file__), 'synthetic_data_science_scores.csv')
data = pd.read_csv(data_path)

@score_api.route('/api/percentile', methods=['POST'])
def calculate_percentile():
    scores = request.get_json()
    mcq_score = scores.get('mcq')
    frq_score = scores.get('frq')

    if mcq_score is None or frq_score is None:
        return jsonify({'error': 'Missing mcq or frq scores'}), 400

    mcq_percentile = percentileofscore(data['mcq'], mcq_score, kind='rank')
    frq_percentile = percentileofscore(data['frq'], frq_score, kind='rank')

    return jsonify({
        'mcq_percentile': mcq_percentile,
        'frq_percentile': frq_percentile
    })
