# model/college.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class CollegePredictor:
    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['gpa', 'sat', 'act', 'apCount', 'extracurriculars']
        self.target = 'chance'
        self.college_data = self._load_data()

    def _load_data(self):
        # Simulated training data
        data = {
            'gpa': [3.0, 3.2, 3.5, 3.8, 4.0, 3.9, 3.6, 3.1, 2.9],
            'sat': [1100, 1200, 1300, 1400, 1500, 1480, 1350, 1250, 1000],
            'act': [21, 24, 27, 30, 33, 32, 28, 23, 19],
            'apCount': [2, 3, 5, 7, 10, 9, 6, 4, 1],
            'extracurriculars': [3, 4, 6, 8, 10, 9, 7, 5, 2],
            'chance': [30, 40, 55, 70, 90, 88, 65, 45, 20]
        }
        return pd.DataFrame(data)

    def _train(self):
        X = self.college_data[self.features]
        y = self.college_data[self.target]

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
        return {'predicted_chance': round(prediction, 2)}

    def importance(self):
        importances = self.dt.feature_importances_
        return {feature: round(importance, 2) for feature, importance in zip(self.features, importances)}

def initCollege():
    CollegePredictor.get_instance()

def testCollege():
    print("Step 1: Sample Student Data")
    student = {
        'gpa': 3.7,
        'sat': 1450,
        'act': 31,
        'apCount': 6,
        'extracurriculars': 8
    }
    print("\t", student)
    model = CollegePredictor.get_instance()

    print("Step 2: Predict Acceptance Chance")
    prediction = model.predict(student)
    print("\t Predicted Acceptance Chance:", prediction)

    print("Step 3: Feature Importance")
    importances = model.importance()
    for k, v in importances.items():
        print(f"\t {k}: {v}")

if __name__ == "__main__":
    testCollege()