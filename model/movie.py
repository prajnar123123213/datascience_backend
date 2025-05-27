# movie_predictor.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class MoviePredictor:
    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.features = ['Budget', 'Fame', 'MemePotential']
        self.data = self._load_data()

    def _load_data(self):
        # Simulated dataset for demonstration
        data = pd.DataFrame([
            {'Genre': 'Action', 'Budget': 150, 'Fame': 5, 'MemePotential': 60, 'Hit': 1},
            {'Genre': 'Comedy', 'Budget': 30, 'Fame': 3, 'MemePotential': 80, 'Hit': 1},
            {'Genre': 'Drama', 'Budget': 10, 'Fame': 2, 'MemePotential': 10, 'Hit': 0},
            {'Genre': 'Horror', 'Budget': 5, 'Fame': 1, 'MemePotential': 90, 'Hit': 0},
            {'Genre': 'Sci-Fi', 'Budget': 120, 'Fame': 4, 'MemePotential': 70, 'Hit': 1},
        ])
        return data

    def _clean(self):
        genre_encoded = self.encoder.fit_transform(self.data[['Genre']]).toarray()
        genre_cols = ['Genre_' + g for g in self.encoder.categories_[0]]
        genre_df = pd.DataFrame(genre_encoded, columns=genre_cols)

        self.data = pd.concat([self.data, genre_df], axis=1)
        self.data.drop(['Genre'], axis=1, inplace=True)
        self.features.extend(genre_cols)

    def _train(self):
        X = self.data[self.features]
        y = self.data['Hit']
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, movie):
        movie_df = pd.DataFrame([movie])
        genre_encoded = self.encoder.transform(movie_df[['Genre']]).toarray()
        genre_cols = ['Genre_' + g for g in self.encoder.categories_[0]]
        genre_df = pd.DataFrame(genre_encoded, columns=genre_cols)

        movie_df = pd.concat([movie_df, genre_df], axis=1)
        movie_df.drop(['Genre'], axis=1, inplace=True)

        for col in self.features:
            if col not in movie_df.columns:
                movie_df[col] = 0

        prediction = self.model.predict(movie_df)[0]
        return {"result": "Hit" if prediction == 1 else "Flop"}

    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: round(importance, 3) for feature, importance in zip(self.features, importances)}

def init_movie():
    MoviePredictor.get_instance()

def test_movie():
    movie = {
        'Genre': 'Action',
        'Budget': 100,
        'Fame': 4,
        'MemePotential': 70
    }

    predictor = MoviePredictor.get_instance()
    print(predictor.predict(movie))
    print(predictor.feature_weights())

if __name__ == "__main__":
    test_movie()
