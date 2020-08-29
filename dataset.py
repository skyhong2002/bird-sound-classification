import glob
import json
import os
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np

import env

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(file_dict, batch_size=env.TRAIN.BATCH_SIZE, repeat=True):
    def npy_load_wrapper(item):
        # print(item)
        data = np.load(item.numpy().decode())
        return data

    def open_npy(filename):
        # print(filename)
        d = tf.convert_to_tensor(tf.py_function(npy_load_wrapper, [filename], [tf.float32]))

        return d[0, :, :]
        # return np.load(filename)

    paths = []
    labels = []
    tmp_list = []
    for k, v in file_dict.items():
        for u in v:
            tmp_list.append((u, env.classes[k]))
            # paths.append(u)
            # labels.append(env.classes[k])
    random.shuffle(tmp_list)

    for p, l in tmp_list:
        paths.append(p)
        # assert l < len(env.classes.keys())
        # onehot = [1 if x == l else 0 for x in range(len(env.classes.keys()))]
        labels.append(l)

    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    files_ds = paths_ds.map(open_npy, num_parallel_calls=AUTOTUNE)
    ds = tf.data.Dataset.zip((files_ds, labels_ds))

    ds = ds.shuffle(min(len(paths), 10000))
    if repeat:
        ds = ds.repeat()
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([env.DATASET.MAX_TIME_STEP, 128]),
            tf.TensorShape([]),
            # tf.TensorShape([len(env.classes.keys())])
        ),
        padding_values=(
            env.PADDING_VAL,
            tf.constant(0, dtype=tf.int64)
        )
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds, np.ceil(len(paths) / batch_size)


if __name__ == '__main__':
    split_val = (input("split val set? [y/N]: ").lower() == "y")
    split_test = (input("split val set? [y/N]: ").lower() == "y")
    ds_dir = input(f"Dataset dir? [\"{env.DATASET.DIR}\"]: ") or env.DATASET.DIR
    dataset_train = {}
    dataset_val = {}
    dataset_test = {}
    for dir_name in tqdm(glob.glob(f"{ds_dir}/*/")):
        class_name = os.path.basename(dir_name[:-1])
        ll = sorted(glob.glob(f"{dir_name}/*.npy"),
                    key=lambda y: int(os.path.basename(y).split('.')[0].replace("XC", "")))[::-1]

        # [VAL, ... TEST, ...., TRAIN]
        val_slice_idx = len(ll) // 10 if split_val else 0
        test_slice_idx = val_slice_idx + (len(ll) // 5 if split_test else 0)
        dataset_train[class_name] = ll[max(val_slice_idx, test_slice_idx):]
        dataset_val[class_name] = ll[:val_slice_idx]
        dataset_test[class_name] = ll[val_slice_idx:test_slice_idx]

    print("verifying (1/2)")
    # check no duplicated
    train_list = set()
    for x in tqdm(dataset_train.values()):
        train_list.union(set(x))

    print("verifying (2/2)")
    for i in tqdm(list(dataset_val.values()) + list(dataset_test.values())):
        for j in i:
            if j in train_list:
                raise Exception("duplicated!")

    with open(f"{ds_dir}/list.json", "w") as f:
        json.dump({
            'train': dataset_train,
            'valid': dataset_val,
            'test': dataset_test
        }, f)
    print("Done")
