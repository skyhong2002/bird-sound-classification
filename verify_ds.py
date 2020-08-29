import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


def job(filename):
    a = np.load(filename)
    if np.any(np.isnan(a)):
        print(f"{filename} NaN!")


DATASET_DIR = "dataset"
files = list(glob(f"{DATASET_DIR}/*/*.npy"))

with Pool(64) as p:
    list(tqdm(p.imap(job, files)))
