# api/score.py

from flask import Blueprint, request, jsonify
import pandas as pd
from scipy.stats import percentileofscore
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'synthetic_data_science_scores.csv')
data = pd.read_csv(data_path)

# Score class definition
class Score:
    def __init__(self, mcq, frq):
        self.mcq = mcq
        self.frq = frq

    def mcq_percentile(self):
        return percentileofscore(data['mcq'], self.mcq, kind='rank')

    def frq_percentile(self):
        return percentileofscore(data['frq'], self.frq, kind='rank')

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
