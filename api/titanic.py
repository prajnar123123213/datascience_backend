from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from flask_login import current_user, login_required

from __init__ import db
from model.titanic import TitanicModel  # ML model
from model.titanicprediction import TitanicPrediction  # SQLAlchemy DB model

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class TitanicAPI:
    class _Predict(Resource):
        @login_required
        def post(self):
            passenger = request.get_json()

            # Run prediction
            titanicModel = TitanicModel.get_instance()
            response = titanicModel.predict(passenger)

            # Save prediction to DB
            prediction = TitanicPrediction(
                user_id=current_user.id,
                input_data=passenger,
                prediction_result=response
            )
            prediction.create()  # Save to DB

            return jsonify(response)

    # Only add /predict, not /history
    api.add_resource(_Predict, '/predict')