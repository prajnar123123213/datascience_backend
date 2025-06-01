# model/destination_model.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Destination:
    _instance = None
    @staticmethod
    def get_instance():
        if Destination._instance is None:
            Destination()
        return Destination._instance

    def __init__(self):
        if Destination._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Destination._instance = self
            self._load_model()

    def _load_model(self):
        self.data = [
            ['summer', 'beach', 'low', 'asia', 'bali'],
            ['winter', 'skiing', 'high', 'europe', 'switzerland'],
            ['fall', 'hiking', 'medium', 'north america', 'colorado'],
            ['spring', 'museums', 'medium', 'europe', 'paris'],
            ['summer', 'safari', 'high', 'africa', 'kenya'],
            ['winter', 'northern lights', 'high', 'europe', 'iceland'],
            ['fall', 'wine tasting', 'high', 'south america', 'argentina'],
            ['spring', 'temples', 'low', 'asia', 'thailand'],
            ['summer', 'surfing', 'low', 'oceania', 'australia'],
            ['winter', 'shopping', 'medium', 'asia', 'dubai']
        ]

        X = [row[:4] for row in self.data]
        y = [row[4] for row in self.data]

        self.encoders = [LabelEncoder() for _ in range(4)]
        X_encoded = np.array([
            [self.encoders[i].fit_transform([row[i] for row in X])[j] for i in range(4)]
            for j in range(len(X))
        ])

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.model = DecisionTreeClassifier()
        self.model.fit(X_encoded, y_encoded)

    def predict(self, data):
        try:
            input_row = [data['season'], data['activity'], data['budget'], data['continent']]
            input_encoded = [
                self.encoders[i].transform([input_row[i]])[0] for i in range(4)
            ]
        except (ValueError, KeyError) as e:
            return {'error': f'Invalid input: {e}'}

        pred_encoded = self.model.predict([input_encoded])[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        return {'destination': pred_label}
