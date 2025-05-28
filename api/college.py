# api/college.py

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.college import CollegePredictor

college_api = Blueprint('college_api', __name__, url_prefix='/api/college')
api = Api(college_api)

class CollegeAPI:
    class _Chance(Resource):
        def post(self):
            data = request.get_json()
            model = CollegePredictor.get_instance()
            prediction = model.predict(data)
            return jsonify(prediction)

    class _Importance(Resource):
        def get(self):
            model = CollegePredictor.get_instance()
            weights = model.feature_weights()
            return jsonify(weights)

    # 👇 New endpoint names here:
    api.add_resource(_Chance, '/chance')
    api.add_resource(_Importance, '/importance')
