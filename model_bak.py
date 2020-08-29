import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import backend as K

import env


def position_encoding(time_step_size, channels):
    """
    Position Encoding described in section 4.1 of
    End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).
    Args:
      time_step_size: length of the sentence
      channels: dimensionality of the embeddings
    Returns:
      A numpy array of shape [sentence_size, embedding_size] containing
      the fixed position encodings for each sentence position.
    """

    encoding = np.ones((time_step_size, channels), dtype=np.float32)
    ls = time_step_size + 1
    le = channels + 1
    for k in range(1, le):
        for j in range(1, ls):
            encoding[j - 1, k - 1] = (1.0 - j / float(ls)) - (
                    k / float(le)) * (1. - 2. * j / float(ls))
    return encoding


def conv2d(x, filters, kernel_size=3, norm=True):
    if norm:
        x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same", activation=None, use_bias=False)(x)
        return tf.nn.leaky_relu(layers.BatchNormalization()(x))
    else:
        return layers.Conv2D(filters, kernel_size=kernel_size, padding="same", activation=tf.nn.leaky_relu)(x)


def conv_each_time_step(x, filters, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu, norm=True, dropout=0.5):
    """
    Input:  batch x timeStep x freq x channels
    Output: batch x timeStep x freq x filters
    """
    x = layers.TimeDistributed(
        layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=(not norm)))(x)
    if norm:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    if activation is not None:
        x = activation(x)
    x = layers.Dropout(rate=dropout)(x)
    return x


def max_pool_each_time_step(x, pool_size=2, strides=2, padding='SAME'):
    """
    Input:  batch x timeStep x freq x channels
    Output: batch x timeStep x freq/n x channels
    """

    x = layers.TimeDistributed(
        layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding))(x)
    return x


def avg_pool_each_time_step(x, pool_size=2, strides=2, padding='SAME'):
    """
    Input:  batch x timeStep x freq x channels
    Output: batch x timeStep x freq/n x channels
    """
    x = layers.TimeDistributed(
        layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding))(x)
    return x


def attention_t(x, n_value, n_query):
    """
    input: (BATCH x time_step x channels)
    out  : (BATCH x channels)
    """
    query = layers.Conv1D(filters=n_query, kernel_size=1, strides=1)(x)
    key = layers.Conv1D(filters=n_query, kernel_size=1, strides=1)(x)
    value = layers.Conv1D(filters=n_value, kernel_size=1, strides=1)(x)

    key = tf.transpose(key, perm=[0, 2, 1])  # BATCH x in_channels x N
    s_map = tf.matmul(query, key)  # BATCH x N x N
    s_map = n_query ** -.5 * s_map
    s_map = tf.nn.softmax(s_map, axis=-1)
    context = tf.matmul(s_map, value)  # BATCH x N x 1

    context_t = tf.transpose(context, perm=[0, 2, 1])  # BATCH x n_key x N

    out = tf.matmul(context_t, context) * tf.cast(K.shape(context_t)[-1], dtype=tf.float32) ** -.5

    return out


def attention(x, n_value, n_query):
    query = layers.Conv1D(filters=n_query, kernel_size=1, strides=1)(x)
    key = layers.Conv1D(filters=n_query, kernel_size=1, strides=1)(x)
    value = layers.Conv1D(filters=n_value, kernel_size=1, strides=1)(x)

    key = tf.transpose(key, perm=[0, 2, 1])  # BATCH x channels x N
    s_map = tf.matmul(query, key)  # BATCH x N x N
    s_map = n_query ** -.5 * s_map
    s_map = tf.nn.softmax(s_map, axis=-1)
    context = tf.matmul(s_map, value)

    return context, s_map


def get_model(n_classes=10):
    inputs = layers.Input((None, 64, 1))
    x = inputs
    x = conv_each_time_step(x, 16)
    x = conv_each_time_step(x, 16)
    x = max_pool_each_time_step(x)

    x = conv_each_time_step(x, 32)
    x = conv_each_time_step(x, 32)
    x = max_pool_each_time_step(x)

    x = conv_each_time_step(x, 64)
    x = conv_each_time_step(x, 64)
    x = max_pool_each_time_step(x)

    x = conv_each_time_step(x, 64)
    x = conv_each_time_step(x, 64)
    x = max_pool_each_time_step(x)

    x = conv_each_time_step(x, 64)
    x = conv_each_time_step(x, 64)

    x = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)
    x = x + position_encoding(env.TIME_STEP, 64)
    x, _ = attention(x, 64, 12)
    # TODO self attention
    # x = layers.GRU(units=128)(x)
    # x = layers.LSTM(units=128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(units=n_classes, activation=None)(x)  # n_class

    model = tf.keras.Model(inputs=(inputs,), outputs=[x])
    return model
