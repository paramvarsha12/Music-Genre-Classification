import os
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np


DATASET_PATH = 'spectrograms/'
genres = os.listdir(DATASET_PATH)
labels = []

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.isdir(genre_path):
        continue
    for img_file in os.listdir(genre_path):
        if img_file.endswith('.png'):
            labels.append(genre)

labels = np.array(labels)
label_encoder = LabelEncoder()
label_encoder.fit(labels)

os.makedirs('models', exist_ok=True)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print(" Label encoder saved as models/label_encoder.pkl")
