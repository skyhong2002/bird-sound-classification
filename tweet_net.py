import tensorflow as tf

import env
from modules.encoders import Encoder
from modules.utils import positional_encoding, create_padding_mask
from tensorflow.keras.layers import Input, TimeDistributed, Dense
from tensorflow.keras import Model


def get_model(*args):
    return transformer_based_model(*args)

def transformer_based_model(
        layer_type: str = env.MODEL.LAYER_TYPE,
        n_layers: int = env.MODEL.N_LAYERS,
        d_model: int = env.MODEL.D_MODEL,
        d_ff: int = env.MODEL.D_FF,
        n_heads: int = env.MODEL.N_HEADS,
        n_classes: int = len(env.classes.keys()),
        d_input: int = env.MODEL.D_INPUT,
        max_timesteps: int = env.DATASET.MAX_TIME_STEP,
        return_attentions: bool = False):
    """
    build TweetNet Model.
    Args:
        layer_type: 'Conv1D' or 'FFN'. Layer Type in EncoderLayer.
        n_layers: num of encoder layers
        d_model: dim for attention
        d_ff: dim for EncoderLayer
        n_heads: num heads for mh-attention
        max_timesteps:
        d_dense:
        n_classes: shape of output layer
        d_input:
        return_attentions: return attention map or not. if True, return (out, attention_weights) else return out only

    Returns:
        out:Tensor shaped (BATCH, n_classes)
        attentions: TODO
    """
    assert d_model % n_heads == 0
    encoder = Encoder(layer_type, n_layers, d_model, n_heads, d_ff)
    pos_encoding = positional_encoding(max_timesteps, d_model)
    pos_encoding = tf.concat([tf.zeros_like(pos_encoding[:, 0:1, ...]), pos_encoding], axis=1)

    inp = Input(shape=(None, d_input))
    x = inp
    cls_token = tf.ones_like(x[:, 0:1, ...]) * env.CLS_VAL
    x = tf.concat([cls_token, x], axis=1)  # add <CLS>
    padding_mask = create_padding_mask(x)
    x = TimeDistributed(Dense(units=d_model))(x)
    x += pos_encoding
    x, attentions = encoder(x, mask=padding_mask)  # [BATCH, time_step, 256(D_MODEL)]
    x = x[:, 0, ...]
    # x = Dense(units=d_dense, activation="relu")(x)
    x = Dense(units=n_classes)(x)
    if return_attentions:
        model = Model(inputs=inp, outputs=(x, attentions, padding_mask))
    else:
        model = Model(inputs=inp, outputs=x)
    return model


@DeprecationWarning
def legacy_model():
    raise NotImplementedError()
