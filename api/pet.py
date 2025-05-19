# api/pet.py

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.pet import Pet

pet_api = Blueprint('pet_api', __name__, url_prefix='/api/pet')
api = Api(pet_api)

class PetAPI:
    class _Predict(Resource):
        def post(self):
            data = request.get_json()
            pet_model = Pet.get_instance()
            prediction = pet_model.predict(data)
            return jsonify(prediction)

    class _FeatureWeights(Resource):
        def get(self):
            pet_model = Pet.get_instance()
            weights = pet_model.feature_weights()
            return jsonify(weights)

    api.add_resource(_Predict, '/predict')
    api.add_resource(_FeatureWeights, '/features')
