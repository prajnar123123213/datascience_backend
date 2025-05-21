# model/score.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Score:
    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['study', 'sleep', 'effort']
        self.target = 'score'
        self.score_data = self._load_data()

    def _load_data(self):
        # Simulated training data
        data = {
            'study': [1, 2, 3, 4, 5, 2, 1, 3, 4],
            'sleep': [5, 4, 3, 2, 1, 3, 4, 1, 5],
            'effort': [2, 2, 3, 4, 5, 5, 4, 3, 1],
            'score': [6, 7, 8, 9, 9, 6, 5, 6, 8]
        }
        return pd.DataFrame(data)

    def _train(self):
        X = self.score_data[self.features]
        y = self.score_data[self.target]

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
        return {'predicted_score': round(prediction, 2)}

    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: round(importance, 2) for feature, importance in zip(self.features, importances)}

def initScore():
    Score.get_instance()

def testScore():
    print(" Step 1: Sample Student Data")
    student = {'study': 4, 'sleep': 3, 'effort': 4}
    print("\t", student)
    model = Score.get_instance()

    print(" Step 2: Predict Score")
    prediction = model.predict(student)
    print("\t Predicted Score:", prediction)

    print(" Step 3: Feature Importance")
    importances = model.feature_weights()
    for k, v in importances.items():
        print(f"\t {k}: {v}")

if __name__ == "__main__":
    testScore()
