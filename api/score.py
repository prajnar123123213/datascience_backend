# api/score.py

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.score import Score

score_api = Blueprint('score_api', __name__, url_prefix='/api/score')
api = Api(score_api)

class ScoreAPI:
    class _Predict(Resource):
        def post(self):
            data = request.get_json()
            score_model = Score.get_instance()
            prediction = score_model.predict(data)
            return jsonify(prediction)

    class _FeatureWeights(Resource):
        def get(self):
            score_model = Score.get_instance()
            weights = score_model.feature_weights()
            return jsonify(weights)

    api.add_resource(_Predict, '/predict')
    api.add_resource(_FeatureWeights, '/features')
