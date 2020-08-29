import json
import os
import sys
from threading import Thread

from tensorflow.python.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import env
import model
from dataset import get_dataset

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    m = model.get_model(n_classes=len(env.classes.keys()))
    print("preparing datasets..", flush=True)
    with open("dataset/list.json") as f:
        ds = json.load(f)
        train_set, valid_set = ds['train'], ds['valid']
        train_ds, ts = get_dataset(train_set, env.BATCH_SIZE)
        valid_ds, vs = get_dataset(valid_set, env.BATCH_SIZE)

    checkpoint = ModelCheckpoint("checkpoints/ckpt",
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # Sparse
    m.compile(
        loss=SparseCategoricalCrossentropy(
            from_logits=True,
        ), optimizer=opt, metrics=['accuracy'])

    print("start training..", flush=True)
    m.fit(
        train_ds, validation_data=valid_ds,
        steps_per_epoch=ts, validation_steps=vs,
        callbacks=[checkpoint],
        epochs=100
    )


if __name__ == '__main__':
    main()
