from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

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
        # [season, activity, budget, continent, destination]
        self.data = [
            ['summer', 'beach', 'low', 'asia', 'bali'],
            ['winter', 'skiing', 'high', 'europe', 'switzerland'],
            ['autumn', 'hiking', 'medium', 'north america', 'colorado'],
            ['spring', 'museums', 'medium', 'europe', 'paris'],
            ['summer', 'safari', 'high', 'africa', 'kenya'],
            ['winter', 'northern lights', 'high', 'europe', 'iceland'],
            ['autumn', 'wine tasting', 'high', 'south america', 'argentina'],
            ['spring', 'temples', 'low', 'asia', 'thailand'],
            ['summer', 'surfing', 'low', 'oceania', 'australia'],
            ['winter', 'shopping', 'medium', 'asia', 'dubai'],
            ['summer', 'surfing', 'medium', 'oceania', 'new zealand'],
            ['spring', 'hot springs', 'high', 'asia', 'japan'],
            ['winter', 'skiing', 'medium', 'north america', 'canada'],
            ['autumn', 'wine tasting', 'medium', 'europe', 'italy'],
            ['spring', 'museums', 'low', 'europe', 'greece'],
            ['summer', 'island hopping', 'medium', 'asia', 'philippines'],
            ['winter', 'shopping', 'high', 'asia', 'singapore'],
            ['autumn', 'hiking', 'low', 'south america', 'peru'],
            ['spring', 'temples', 'medium', 'asia', 'cambodia'],
            ['summer', 'beach', 'high', 'north america', 'bahamas'],
            ['winter', 'northern lights', 'medium', 'north america', 'alaska'],
            ['autumn', 'hot springs', 'low', 'europe', 'hungary'],
            ['spring', 'safari', 'high', 'africa', 'south africa'],
            ['summer', 'wine tasting', 'medium', 'south america', 'chile'],
            ['winter', 'museums', 'medium', 'europe', 'germany'],
            ['autumn', 'shopping', 'medium', 'asia', 'south korea'],
            ['spring', 'surfing', 'low', 'oceania', 'fiji'],
            ['summer', 'temples', 'low', 'asia', 'vietnam'],
            ['autumn', 'spa', 'high', 'europe', 'sweden'],
            ['winter', 'beach', 'medium', 'africa', 'seychelles'],
            ['spring', 'hiking', 'medium', 'europe', 'switzerland'],
            ['summer', 'safari', 'medium', 'africa', 'tanzania'],
            ['autumn', 'museums', 'low', 'north america', 'washington dc'],
            ['spring', 'island hopping', 'medium', 'oceania', 'french polynesia'],
            ['winter', 'hot springs', 'high', 'north america', 'yellowstone'],
            ['summer', 'shopping', 'low', 'asia', 'malaysia']
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

        # Activity mapping (frontend activity -> real activity)
        self.activity_map = {
            'adventure': ['skiing', 'surfing', 'hiking', 'safari'],
            'relaxation': ['beach', 'wine tasting', 'hot springs', 'spa'],
            'sightseeing': ['museums', 'northern lights', 'temples', 'shopping'],
            'cultural': ['temples', 'museums', 'wine tasting'],
            'beach': ['beach', 'surfing', 'island hopping'],
            'nature': ['safari', 'hiking', 'northern lights']
        }

        # Mapping real activity to destination (optional helper)
        self.activity_to_destination = {
            row[1]: row[4] for row in self.data
        }

    def predict(self, data):
        try:
            # Use simplified input
            season = data['season']
            activity_category = data['activity']
            budget = data['budget']
            continent = data['continent']

            # Choose a matching real-world activity from the category
            possible_activities = self.activity_map.get(activity_category, [])
            if not possible_activities:
                return {'error': 'Unknown activity category'}

            chosen_activity = random.choice(possible_activities)

            # Predict using the chosen activity
            input_row = [season, chosen_activity, budget, continent]
            input_encoded = [
                self.encoders[i].transform([input_row[i]])[0] for i in range(4)
            ]

            pred_encoded = self.model.predict([input_encoded])[0]
            destination = self.label_encoder.inverse_transform([pred_encoded])[0]

            return {
                'destination': destination.title(),
                'activitySuggestion': f"{chosen_activity.title()} in {destination.title()}"
            }

        except (ValueError, KeyError) as e:
            return {'error': f'Invalid input: {e}'}
