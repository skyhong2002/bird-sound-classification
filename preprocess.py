from multiprocessing import set_start_method, get_context
import logging
import os
import sys
import numpy as np
import glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import env
import librosa
import traceback

warnings.filterwarnings('ignore', message='PySoundFile failed')
TMP_FILE = os.environ.get('TMP_FILE', '/tmp/preprocess.ckpt')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', -1))
N_THREAD = int(os.environ['N_THREAD']) if 'N_THREAD' in os.environ else 4
MIN_TIME_STEP = env.DATASET.MIN_TIME_STEP
TIME_STEP = env.DATASET.MAX_TIME_STEP
# AUDIO_DIR = "/home/ray1422/workspace/tweet_database/data"
AUDIO_DIR = os.environ["AUDIO_DIR"] if 'AUDIO_DIR' in os.environ else "/home/phlee0514/Ray/bird_crawler/data"
# FEATURE = os.environ.get('FEATURE', 'spectrogram').lower()
FEATURE = env.DATASET.FEATURE
OUTPUT_DIR = os.environ.get('OUTPUT_DIR',
                            f'./dataset_{FEATURE}-{env.DATASET.MIN_TIME_STEP}-{env.DATASET.MAX_TIME_STEP}')

logging.basicConfig(level=logging.INFO)
FEATURE = os.environ.get('FEATURE', 'spectrogram').lower()

assert FEATURE in ('mfcc', 'spectrogram')

print(f"length: {MIN_TIME_STEP}-{TIME_STEP}")
print(f"feature: {FEATURE}")
print(f"output dir: {OUTPUT_DIR}")
print(f"shape: {env.MODEL.D_INPUT}")
print(f"PID: {os.getpid()}")


def job(parm):
    try:
        class_name, filename = parm
        # y, sr = librosa.load(filename, sr=None)
        y, sr = librosa.load(filename, sr=16000)
        arr = None
        if FEATURE == "spectrogram":
            # 提取 mel spectrogram feature
            spe = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=env.MODEL.D_INPUT)
            arr = librosa.amplitude_to_db(spe)
        elif FEATURE == "mfcc":
            arr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=env.MODEL.D_INPUT)
        arr = arr.T

        if np.any(np.isnan(arr)):
            logging.error(f"NaN in {filename}")
            return

        # if arr.shape[0] < MIN_TIME_STEP:
        #     logging.info(class_name + " " + os.path.basename(filename) + " is too short, ignored.")
        #     return
        class_name = class_name.replace(" ", "_").replace("'", '-')
        for i, sub_arr in enumerate(np.array_split(arr, np.ceil(arr.shape[0] / TIME_STEP))):
            if sub_arr.shape[0] < MIN_TIME_STEP:
                logging.info(f"{sub_arr.shape[0]} is too short.")
            os.system(f"mkdir -p {OUTPUT_DIR}/{class_name}/")
            np.save(f"{OUTPUT_DIR}/{class_name}/{os.path.basename(filename)}_{i}.npy", sub_arr)
    except Exception as e:
        logging.error("An error occurred: " + str(e))
        return 0


def main():
    process_queue = []
    for dir_name in glob.glob(f"{AUDIO_DIR}/*/"):
        class_name = os.path.basename(dir_name[:-1])
        for filename in glob.glob(f"{dir_name}/*.wav"):
            process_queue.append((class_name, filename))
    chunk_size = CHUNK_SIZE if CHUNK_SIZE > 0 else len(process_queue)
    start = 0
    if os.path.isfile(TMP_FILE):
        with open(TMP_FILE) as f:
            start = int(f.readline().strip())
            print(f"checkpoint found! start from {start}")

    # for parm in process_queue:
    #     job(parm)
    while start < len(process_queue):
        end = start + chunk_size
        print(
            f"{start}/{len(process_queue)} ({start // chunk_size + 1}/{int(len(process_queue) // chunk_size)})")
        with Pool(N_THREAD) as p:
            _ = list(tqdm(p.imap_unordered(job, process_queue[start:end]), file=sys.stdout, total=chunk_size))

        with open(TMP_FILE, "w") as f:
            f.write(f"{end}")
        start = end

    print("Done!")


if __name__ == '__main__':
    main()
