# MoviePredictor Model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class MoviePredictor:
    """A class for predicting whether a movie will be a Hit or Flop based on input features."""

    _instance = None  # Singleton instance

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['Genre', 'Budget', 'Fame', 'MemePotential']
        self.movie_data = None

    def _load_data(self):
        # Manually constructed dummy dataset (could be replaced by real data)
        self.movie_data = pd.DataFrame([
            {'Genre': 'Action', 'Budget': 100, 'Fame': 5, 'MemePotential': 80, 'BoxOffice': 1},
            {'Genre': 'Comedy', 'Budget': 30, 'Fame': 3, 'MemePotential': 60, 'BoxOffice': 1},
            {'Genre': 'Drama', 'Budget': 10, 'Fame': 2, 'MemePotential': 20, 'BoxOffice': 0},
            {'Genre': 'Action', 'Budget': 150, 'Fame': 4, 'MemePotential': 90, 'BoxOffice': 1},
            {'Genre': 'Romance', 'Budget': 20, 'Fame': 1, 'MemePotential': 10, 'BoxOffice': 0},
            {'Genre': 'Horror', 'Budget': 15, 'Fame': 2, 'MemePotential': 50, 'BoxOffice': 0},
            {'Genre': 'Sci-Fi', 'Budget': 120, 'Fame': 4, 'MemePotential': 70, 'BoxOffice': 1},
            {'Genre': 'Comedy', 'Budget': 25, 'Fame': 3, 'MemePotential': 65, 'BoxOffice': 1},
            {'Genre': 'Drama', 'Budget': 8, 'Fame': 2, 'MemePotential': 15, 'BoxOffice': 0},
            {'Genre': 'Action', 'Budget': 200, 'Fame': 5, 'MemePotential': 95, 'BoxOffice': 1}
        ])

        # One-hot encode the 'Genre' column
        genre_encoded = pd.get_dummies(self.movie_data['Genre'], prefix='Genre')
        self.movie_data = pd.concat([self.movie_data.drop('Genre', axis=1), genre_encoded], axis=1)

        # Save the one-hot genre column names for future predictions
        self.genre_columns = genre_encoded.columns.tolist()

    def _train(self):
        # Features and target
        X = self.movie_data.drop('BoxOffice', axis=1)
        y = self.movie_data['BoxOffice']

        # Train logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X, y)

        # Also train decision tree for feature importances
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

        # Save the feature list after encoding
        self.features = X.columns.tolist()

    @classmethod
    def get_instance(cls):
        """Gets or creates a singleton instance of MoviePredictor"""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_data()
            cls._instance._train()
        return cls._instance

    def predict(self, movie_input):
        """
        Predict if the movie will be a Hit or Flop.
        Args:
            movie_input (dict): Should include 'Genre', 'Budget', 'Fame', 'MemePotential'
        Returns:
            dict: {'result': 'Hit' or 'Flop'}
        """
        df = pd.DataFrame([movie_input])

        # One-hot encode Genre like training data
        for col in self.genre_columns:
            df[col] = 0
        genre_col = f"Genre_{movie_input['Genre']}"
        if genre_col in self.genre_columns:
            df[genre_col] = 1

        df.drop('Genre', axis=1, inplace=True)

        # Fill in any missing columns (in case future training set changes)
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match training set
        df = df[self.features]

        # Predict
        result = self.model.predict(df)[0]
        return {'result': 'Hit' if result == 1 else 'Flop'}

    def feature_weights(self):
        """Return the importance of each feature from the Decision Tree"""
        importances = self.dt.feature_importances_
        return {feature: float(f"{imp:.3f}") for feature, imp in zip(self.features, importances)}
