raise DeprecationWarning

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import env
from modules.encoders import *

N_LAYERS = 2
N_HEADS = 4
DFF = 128
D_MODEL = 256


def get_model(n_classes=10, return_attentions=False):
    encoder = Encoder(N_LAYERS, D_MODEL, N_HEADS, DFF, env.MAX_TIME_STEP)
    pos_encoding = positional_encoding(env.MAX_TIME_STEP, D_MODEL)  # +1 for <CLS> token
    pos_encoding = tf.concat([tf.zeros_like(pos_encoding[:, 0:1, ...]), pos_encoding], axis=1)
    inp = Input(shape=(None, 128))
    x = inp
    cls_token = tf.ones_like(x[:, 0:1, ...]) * env.CLS_VAL
    x = tf.concat([cls_token, x], axis=1)
    padding_mask = create_padding_mask(x)
    x = TimeDistributed(Dense(units=D_MODEL))(x)
    x += pos_encoding
    x, attentions = encoder(x, mask=padding_mask)  # [BATCH, time_step, 256(D_MODEL)]
    x = x[:, 0, ...]
    # x = tf.reduce_sum(x, axis=1) / tf.reduce_sum(1 - padding_mask[:, 0, 0, :])
    # x = GRU(units=256, activation="tanh")(x, mask=tf.cast(padding_mask[:, 0, 0, :], dtype=tf.bool))
    x = Dense(units=256, activation="relu")(x)
    x = Dense(units=n_classes)(x)
    if return_attentions:
        model = Model(inputs=inp, outputs=(x, attentions, padding_mask))
    else:
        model = Model(inputs=inp, outputs=x)
    return model
