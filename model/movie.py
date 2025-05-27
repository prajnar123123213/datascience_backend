# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import pickle

np.random.seed(42)

# Synthetic data generation
genres = ['Action', 'Comedy', 'Horror', 'Drama', 'Sci-Fi']
fame_levels = [1, 2, 3, 4, 5]  # 1 = unknown actor, 5 = mega star

data = []
for _ in range(500):
    genre = np.random.choice(genres)
    budget = np.random.randint(1, 300)  # in millions
    fame = np.random.choice(fame_levels)
    meme_potential = np.random.randint(0, 101)
    
    # Outcome logic
    score = (budget * 0.3) + (fame * 10) + (meme_potential * 0.6)
    if score > 180:
        outcome = 'Blockbuster'
    elif score > 100:
        outcome = 'Average'
    else:
        outcome = 'Flop'
    
    data.append([genre, budget, fame, meme_potential, outcome])

# Create DataFrame
df = pd.DataFrame(data, columns=['Genre', 'Budget', 'Fame', 'MemePotential', 'Outcome'])

# Split X and y
X = df[['Genre', 'Budget', 'Fame', 'MemePotential']]
y = df['Outcome']

# Preprocess & train
preprocessor = ColumnTransformer(
    transformers=[('genre', OneHotEncoder(), ['Genre'])],
    remainder='passthrough'
)

model = make_pipeline(preprocessor, RandomForestClassifier())
model.fit(X, y)

# Save model
with open('movie_model.pkl', 'wb') as f:
    pickle.dump(model, f)
