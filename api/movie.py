## Python Movie Sample API endpoint
from flask import Blueprint, request, jsonify
from model.movie import MoviePredictor  # Import the MoviePredictor class
from flask_restful import Api, Resource  # used for REST API building

# Set up the Flask Blueprint for the movie predictor
movie_api = Blueprint('movie_api', __name__, url_prefix='/api/movie')

api = Api(movie_api)

class MovieAPI:
    class _Guess(Resource):
        def post(self):
            """POST method to receive movie data and return box office prediction.

            This endpoint receives movie data in JSON format:
              - Genre (e.g., 'Action', 'Comedy', etc.)
              - Budget (in millions)
              - Fame (1–5 scale for actor fame)
              - MemePotential (0–100)
            
            Returns:
                JSON object with result key: 'Hit' or 'Flop'
            """
            # Get the movie data from the request
            movie = request.get_json()

            # Get the trained movie prediction model (singleton)
            model = MoviePredictor.get_instance()

            # Predict the outcome
            prediction = model.predict(movie)

            # Return the prediction
            return jsonify(prediction)

    # Add /guess endpoint to API
    api.add_resource(_Guess, '/guess')
