# api/destination.py

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.destination import Destination  # Import the DestinationModel class

destination_api = Blueprint('destination_api', __name__, url_prefix='/api/destination')
api = Api(destination_api)

class DestinationAPI:
    class _Recommend(Resource):
        def post(self):
            data = request.get_json()
            model = Destination.get_instance()
            recommendation = model.predict(data)
            return jsonify(recommendation)

    api.add_resource(_Recommend, '/recommend')
