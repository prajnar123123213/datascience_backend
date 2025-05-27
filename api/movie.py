from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from movie_predictor import MoviePredictor  # Import your model class

# Create a Blueprint for the movie API
movie_api = Blueprint('movie_api', __name__, url_prefix='/api/movie')

# Initialize Flask-RESTful API on this blueprint
api = Api(movie_api)

class MovieAPI:
    class _Predict(Resource):
        def post(self):
            """
            POST request to predict movie success.
            Accepts a JSON payload with movie data.
            Example input:
            {
                "Genre": "Action",
                "Budget": 100,
                "Fame": 4,
                "MemePotential": 70
            }
            """
            movie = request.get_json()
            movieModel = MoviePredictor.get_instance()
            response = movieModel.predict(movie)
            return jsonify(response)

    api.add_resource(_Predict, '/predict')
