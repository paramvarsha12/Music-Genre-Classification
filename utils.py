import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def audio_to_melspectrogram_image(audio_path, save_path='temp.png'):
    y, sr = librosa.load(audio_path, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel)

    plt.figure(figsize=(2.56, 2.56), dpi=50)  # Creates a 128x128 px image
    plt.axis('off')
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def load_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
