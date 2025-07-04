import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


IMG_SIZE = (128, 128)
DATASET_PATH = 'spectrograms/'


def load_data():
    genres = os.listdir(DATASET_PATH)
    images = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        for img_file in os.listdir(genre_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(genre_path, img_file)
                img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(genre)

    return np.array(images), np.array(labels)

X, y = load_data()
print("Data loaded. Shape of X:", X.shape)
print("Labels:", np.unique(y))


X = X / 255.0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test))


os.makedirs('models', exist_ok=True)
model.save('models/music_genre_cnn.h5')
