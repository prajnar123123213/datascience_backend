# MoviePredictor Model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    """A class for recommending movies based on user preferences."""
    _instance = None

    def __init__(self):
        self.model = None
        self.movie_data = None
        self.features = ['genre_encoded', 'actor_encoded', 'duration']
        self._load_data()

    def _load_data(self):
        """Load and preprocess the movie dataset."""
        self.movie_data = pd.read_csv('movies.csv')  # Must include: title, genre, actor, duration

        # Encode genres and actors as integers
        self.movie_data['genre_encoded'] = self.movie_data['genre'].astype('category').cat.codes
        self.movie_data['actor_encoded'] = self.movie_data['actor'].astype('category').cat.codes

    def _train(self):
        """Train the Nearest Neighbors model on the movie dataset."""
        X = self.movie_data[self.features]
        self.model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.model.fit(X)

    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of the MovieRecommender model.
        
        Returns:
            MovieRecommender: The trained model instance.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._train()
        return cls._instance

    def recommend(self, user_preferences):
        """Recommend movies based on user input.

        Args:
            user_preferences (dict): Dictionary with keys 'genre', 'actor', 'duration'.

        Returns:
            list: List of recommended movie titles.
        """
        # Encode user input
        genre_code = self.movie_data['genre'].astype('category').cat.categories.get_loc(user_preferences['genre'])
        actor_code = self.movie_data['actor'].astype('category').cat.categories.get_loc(user_preferences['actor'])
        duration = user_preferences['duration']

        user_vector = np.array([[genre_code, actor_code, duration]])

        # Find nearest neighbors
        distances, indices = self.model.kneighbors(user_vector)
        recommended_titles = self.movie_data.iloc[indices[0]]['title'].tolist()
        return recommended_titles

def initMovie():
    """Initialize the movie recommender model."""
    MovieRecommender.get_instance()

def testMovie():
    """Test the movie recommender model by providing example user preferences."""
    print(" Step 1: Define user preferences for movie recommendation:")
    user = {
        'genre': 'Action',
        'actor': 'Tom Cruise',
        'duration': 120
    }
    print("\t", user)
    print()

    recommender = MovieRecommender.get_instance()
    print(" Step 2:", recommender.get_instance.__doc__)
    
    print(" Step 3:", recommender.recommend.__doc__)
    recommendations = recommender.recommend(user)
    print("\tRecommended movies:")
    for title in recommendations:
        print("\t -", title)

if __name__ == "__main__":
    print(" Begin:", testMovie.__doc__)
    testMovie()
