import glob
import os
import random

import librosa
import numpy as np
import time
import env


def main():
    for folder in glob.glob("dataset/raw/*/"):
        folder_class = folder.split("/")[-2]
        assert folder_class in env.classes
        queue = []
        for filename in glob.glob(os.path.join(folder, "*.wav")):
            y, sr = librosa.load(filename, sr=None)
            arr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)  # 1 sec ~= 87
            arr = arr.T
            arr = arr[np.shape(arr)[0] % env.n_time_step:, :]
            if arr.shape[0] < env.n_time_step:
                print(filename, "arr is too short, ignored.", arr.shape)
                continue
            for sub_arr in np.split(arr, arr.shape[0] // env.n_time_step):
                queue.append(sub_arr)
        split_idx = max(1, int(len(queue) * env.test_data_ratio)) 
        test_arr = queue[:split_idx]
        train_arr = queue[split_idx:]
        random.shuffle(queue)
        time.sleep(.1)
        test_npy_filename = os.path.join("./dataset", "spectrogram", "test", folder_class + ".npy")
        train_npy_filename = os.path.join("./dataset", "spectrogram", "train", folder_class + ".npy")

        np.save(test_npy_filename, np.asarray(test_arr))  # saved shape: batch, timeStep, freq
        np.save(train_npy_filename, np.asarray(train_arr))  # saved shape: batch, timeStep, freq

        print(test_npy_filename, "saved. length:", len(test_arr) * env.n_time_step)
        print(train_npy_filename, "saved. length:", len(train_arr) * env.n_time_step)


if __name__ == '__main__':
    main()
