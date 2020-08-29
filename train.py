import json
import os
import sys
from threading import Thread

from tensorflow.python.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
import env
import tweet_net
from dataset import get_dataset

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    m = tweet_net.transformer_based_model()
    print("preparing datasets..", flush=True)
    with open(f"{env.DATASET.DIR}/list.json") as f:
        ds = json.load(f)
        train_set, valid_set = ds['train'], ds['valid']
        train_ds, ts = get_dataset(train_set, env.TRAIN.BATCH_SIZE)
        valid_ds, vs = get_dataset(valid_set, env.TRAIN.BATCH_SIZE)
    
    os.system(f"mkdir -p {env.OUTPUT_DIR}/checkpoints")
    checkpoint = ModelCheckpoint(f"{env.OUTPUT_DIR}/checkpoints/ckpt",
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
    # Sparse
    m.compile(
        loss=SparseCategoricalCrossentropy(
            from_logits=True,
        ), optimizer=opt, metrics=['accuracy'])

    print("start training..", flush=True)
    m.fit(
        train_ds, validation_data=valid_ds,
        steps_per_epoch=ts, validation_steps=vs,
        callbacks=[
            TensorBoard(log_dir=f"{env.OUTPUT_DIR}/logs",
                        write_graph=False),
            EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=1000,
                          verbose=1),
            checkpoint,
            ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=1e-5, verbose=1)
        ],
        epochs=10000
    )


if __name__ == '__main__':
    main()
