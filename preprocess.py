import logging
import os
import sys

import librosa
import numpy as np
import glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings

import env

warnings.filterwarnings('ignore')

N_THREAD = int(os.environ['N_THREAD']) if 'N_THREAD' in os.environ else 4
MIN_TIME_STEP = env.MIN_TIME_STEP or 450
TIME_STEP = env.TIME_STEP or 900
# AUDIO_DIR = "/home/ray1422/workspace/tweet_database/data"
AUDIO_DIR = os.environ["AUDIO_DIR"] if 'AUDIO_DIR' in os.environ else "/home/phlee0514/Ray/bird_crawler/data"
OUTPUT_DIR = "./dataset"

logging.basicConfig(level=logging.INFO)


def job(parm):
    try:
        class_name, filename = parm
        # y, sr = librosa.load(filename, sr=None)
        y, sr = librosa.load(filename, sr=16000)
        # 提取 mel spectrogram feature
        spe = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
        arr = librosa.amplitude_to_db(spe)
        # arr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)  # 1 sec ~= 87
        arr = arr.T
        if np.any(np.isnan(arr)):
            logging.error(f"NaN in {filename}")
            return

        if arr.shape[0] < MIN_TIME_STEP:
            logging.info(class_name + " " + os.path.basename(filename) + " is too short, ignored.")
            return
        for i, sub_arr in enumerate(np.array_split(arr, np.ceil(arr.shape[0] / TIME_STEP))):
            if sub_arr.shape[0] < MIN_TIME_STEP:
                continue
            os.system(f"mkdir -p {OUTPUT_DIR}/{class_name}/")
            np.save(f"{OUTPUT_DIR}/{class_name}/{os.path.basename(filename)}_{i}.npy", sub_arr)
    except Exception as e:
        logging.error("An error occurred: " + str(e))


def main():
    process_queue = []
    for dir_name in glob.glob(f"{AUDIO_DIR}/*/"):
        class_name = os.path.basename(dir_name[:-1])
        for filename in glob.glob(f"{dir_name}/*.mp3"):
            process_queue.append((class_name, filename))

        # for parm in process_queue:
        #     job(parm)
    with Pool(N_THREAD) as p:
        _ = list(tqdm(p.imap(job, process_queue), file=sys.stdout, total=len(process_queue)))


if __name__ == '__main__':
    main()
