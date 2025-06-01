from flask import Blueprint, request
from model.movie import MovieRecommender
from flask_restful import Api, Resource

movie_api = Blueprint('movie_api', __name__, url_prefix='/api/movie')
api = Api(movie_api)

class Recommend(Resource):
    def post(self):
        data = request.get_json()
        model = MovieRecommender.get_instance()
        recommendations = model.recommend(data)
        return {'movies': recommendations}

api.add_resource(Recommend, '/recommend')
