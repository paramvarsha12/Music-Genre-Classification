import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


file_path = 'data/genres_original/rock/rock.00000.wav'
y, sr = librosa.load(file_path, duration=30)


mel = librosa.feature.melspectrogram(y=y, sr=sr)
log_mel = librosa.power_to_db(mel)


save_dir = 'spectrograms/rock'
os.makedirs(save_dir, exist_ok=True)


save_path = os.path.join(save_dir, 'rock.00000.png')

plt.figure(figsize=(2, 2))  
librosa.display.specshow(log_mel, sr=sr)
plt.axis('off')            
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f" Saved spectrogram to: {save_path}")
