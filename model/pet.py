# model/pet.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Pet:
    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['food', 'play', 'sleep']
        self.target = 'happiness'
        self.pet_data = self._load_data()

    def _load_data(self):
        # Simulated training data
        data = {
            'food': [1, 2, 3, 4, 5, 2, 1, 3, 4],
            'play': [5, 4, 3, 2, 1, 3, 4, 1, 5],
            'sleep': [2, 2, 3, 4, 5, 5, 4, 3, 1],
            'happiness': [6, 7, 8, 9, 9, 6, 5, 6, 8]
        }
        return pd.DataFrame(data)

    def _train(self):
        X = self.pet_data[self.features]
        y = self.pet_data[self.target]

        self.model = LinearRegression()
        self.model.fit(X, y)

        self.dt = DecisionTreeRegressor()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._train()
        return cls._instance

    def predict(self, data):
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)[0]
        return {'predicted_happiness': round(prediction, 2)}

    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: round(importance, 2) for feature, importance in zip(self.features, importances)}

def initPet():
    Pet.get_instance()

def testPet():
    print(" Step 1: Sample Pet Data")
    pet = {'food': 3, 'play': 4, 'sleep': 3}
    print("\t", pet)
    model = Pet.get_instance()

    print(" Step 2: Predict Happiness")
    prediction = model.predict(pet)
    print("\t Predicted Happiness:", prediction)

    print(" Step 3: Feature Importance")
    importances = model.feature_weights()
    for k, v in importances.items():
        print(f"\t {k}: {v}")

if __name__ == "__main__":
    testPet()
