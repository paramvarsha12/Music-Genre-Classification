import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_DATASET_PATH = 'data/genres_original'
SAVE_PATH = 'spectrograms'

def create_spectrogram(file_path, save_path):
    y, sr = librosa.load(file_path, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel)

    plt.figure(figsize=(2, 2))
    librosa.display.specshow(log_mel, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_all_spectrograms():
    for genre in os.listdir(AUDIO_DATASET_PATH):
        genre_path = os.path.join(AUDIO_DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        genre_save_path = os.path.join(SAVE_PATH, genre)
        os.makedirs(genre_save_path, exist_ok=True)

        for filename in os.listdir(genre_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_path, filename)
                save_file = filename.replace('.wav', '.png')
                save_path = os.path.join(genre_save_path, save_file)

                try:
                    create_spectrogram(file_path, save_path)
                    print(f"Saved: {save_path}")
                except Exception as e:
                    print(f"Error with {file_path}: {e}")

if __name__ == '__main__':
    generate_all_spectrograms()
