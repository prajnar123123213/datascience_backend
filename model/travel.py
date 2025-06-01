# travel_model.py

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os

class TravelModel:
    """A singleton class for travel destination recommendation based on user preferences."""

    _instance = None

    def __init__(self):
        self.model = None
        self.encoders = {}
        self.features = ['season', 'activity', 'budget', 'continent']
        self.target = 'destination'
        self._load_and_prepare_data()
        self._train_model()

    def _load_and_prepare_data(self):
        """Load and preprocess the travel recommendation dataset."""
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'travel_recommendation.csv')
        df = pd.read_csv(file_path)

        # Ensure required columns are present
        required_columns = self.features + [self.target]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file.")

        # Encode categorical variables
        for col in required_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        self.df = df

    def _train_model(self):
        """Train the K-Nearest Neighbors model."""
        X = self.df[self.features]
        y = self.df[self.target]
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(X, y)

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the TravelModel."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, user_input):
        """
        Predict the recommended travel destination based on user preferences.

        Args:
            user_input (dict): A dictionary with keys 'season', 'activity', 'budget', 'continent'.

        Returns:
            dict: A dictionary containing the recommended destination.
        """
        # Encode user input
        encoded_input = []
        for feature in self.features:
            le = self.encoders[feature]
            encoded_value = le.transform([user_input[feature]])[0]
            encoded_input.append(encoded_value)

        # Predict destination
        encoded_prediction = self.model.predict([encoded_input])[0]
        destination = self.encoders[self.target].inverse_transform([encoded_prediction])[0]
        return {'recommended_destination': destination}
