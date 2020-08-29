import librosa
import numpy as np
from matplotlib import cm, pyplot as plt

FILENAME = "../tweet_database/data/115-Regulus_regulus/575455.mp3"
y, sr = librosa.load(FILENAME, sr=16000)
# 提取 mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)
print(np.min(logmelspec), np.max(logmelspec))
plt.imshow(logmelspec[:, :50], origin='lower')
plt.show()
