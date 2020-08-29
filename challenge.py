import logging
import os
import traceback
import warnings
from glob import glob
from multiprocessing import Process, Pool, Manager, Queue
from functools import partial
import librosa
import time
import numpy as np
from tweet_net import transformer_based_model as get_model
import tensorflow as tf
from tqdm import tqdm
import env
import csv

warnings.filterwarnings('ignore', message='PySoundFile failed')
KILL = 'KILL'
WORK_DIR = os.environ.get("WORK_DIR", "")
N_THREAD = int(os.environ.get("N_THREAD", "8"))
SR = 16000
CHUNK_SIZE = 5 * SR
BATCH_SIZE = 64

assert os.path.isdir(WORK_DIR) or os.path.isfile(WORK_DIR)


class Sample:
    def __init__(self, filename: str, index: int, data: np.ndarray):
        self.filename: str = filename
        self.index = index
        self.data: np.ndarray = data


def preprocess_audio_job(job_queue: Queue, audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=SR)
        i = 0
        while i * CHUNK_SIZE < len(y):
            y_i = y[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
            spectrogram = librosa.feature.melspectrogram(y_i, sr, n_fft=1024, hop_length=512, n_mels=env.MODEL.D_INPUT)
            spectrogram = librosa.amplitude_to_db(spectrogram)
            job_queue.put(Sample(audio_file, i, spectrogram))
            i += 1

    except Exception as e:
        logging.error("pre-processing failed due to " + str(e))


def model_daemon(job_queue: Queue, out_queue: Queue):
    m = get_model()
    try:
        m.load_weights(f"{env.OUTPUT_DIR}/checkpoints/ckpt")
    except Exception as e:
        logging.error(f"Fail to load weights! {str(e)}")
    while True:
        try:
            killed = False
            batch_left = BATCH_SIZE
            batch = []
            for i in range(BATCH_SIZE):
                while job_queue.qsize() == 0:
                    pass
                data = job_queue.get()
                if data == KILL:
                    killed = True
                    break                
                batch.append(data)

            if len(batch) == 0:
                break

            batch_feed = []
            item: Sample
            for item in batch:
                data_padded = np.ones((env.DATASET.MAX_TIME_STEP, env.MODEL.D_INPUT)) * env.PADDING_VAL
                data_padded[:item.data.shape[1], :] = item.data.T
                batch_feed.append(data_padded)

            # print("feed:", np.asarray(batch_feed).shape)
            results = m.predict(np.asarray(batch_feed))
            # print("results", results)
            for i in range(results.shape[0]):
                # print(results[i])
                batch[i].data = results[i]
                out_queue.put(batch[i])

            if killed:
                break
        except Exception as e:
            logging.error(f"An error occurred in model daemon: {str(e)}")
    print("Model daemon done!")
    # Notify writer daemon done
    out_queue.put(KILL)


def writer_daemon(out_queue: Queue):
    print("writer start!")
    with open(f"{WORK_DIR}/species_names_to_codes.txt") as f:
        lines = f.readlines()
    keys = {}
    for line in lines:
        line = line.strip()
        class_name, species_id = line.split(",")
        try:
            class_name = class_name.replace(" ", "_").replace("'", '-')

            keys[env.classes[class_name]] = species_id
        except KeyError as e:
            logging.error(f"{class_name} not found in env!")
    with open(f"{WORK_DIR}/results.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        while True:
            try:
                while out_queue.qsize() < 1:
                    pass
                data: Sample = out_queue.get()
                if data == KILL:
                    break

                predict_id = tf.argmax(data.data).numpy()
                class_predict = keys[predict_id]
                media_id = os.path.basename(data.filename).split("_")[0].replace(".mp3", "").replace(".wav", "")
                start_time = 5 * data.index
                end_time = start_time + 5
                start_time_str = f"{str(start_time // (60 * 60)).zfill(2)}:{str((start_time // 60) % 60).zfill(2)}:{str(start_time % 60).zfill(2)}"
                end_time_str = f"{str(end_time // (60 * 60)).zfill(2)}:{str((end_time // 60) % 60).zfill(2)}:{str(end_time % 60).zfill(2)}"
                time_str = f"{start_time_str}-{end_time_str}"
                result = [media_id, time_str, class_predict, '1.0']
                # print(result)
                writer.writerow(result)
                f.flush()
            except Exception as e:
                logging.error(f"An error occurred in writer daemon: {str(e)}")
    print("Writer daemon done!")


def main():
    audios = glob(f"{WORK_DIR}/audio/*")
    m = Manager()
    job_queue = m.Queue()
    out_queue = m.Queue()
    model_process = Process(target=model_daemon, args=(job_queue, out_queue))
    writer_process = Process(target=writer_daemon, args=(out_queue,))
    model_process.start()
    writer_process.start()
    paj_func = partial(preprocess_audio_job, job_queue)
    with Pool(N_THREAD) as p:
        list(tqdm(p.imap_unordered(func=paj_func, iterable=audios), total=len(audios)))

    job_queue.put(KILL)
    model_process.join()
    writer_process.join()


if __name__ == '__main__':
    main()
