import json
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO)
conf_file = os.environ.get('CONF_FILE', 'conf.yaml')
logging.info(f"use conf {conf_file}")
with open(conf_file) as f:
    conf = yaml.safe_load(f) or {}


class DATASET:
    """ DATA """
    # 1 second = 32 time steps
    _data = conf.get('dataset', {})
    DIR = _data.get("dir", './dataset')
    FEATURE = _data.get("feature", 'spectrogram')
    MAX_TIME_STEP = _data.get('max_time_step', 640)
    MIN_TIME_STEP = _data.get('min_time_step', 320)


class TRAIN:
    """ Training Parm """
    _train = conf.get('train', {})
    BATCH_SIZE = _train.get('batch_size', 32)


class MODEL:
    """ Model Architecture """
    _model = conf.get('model', {})
    D_INPUT = _model.get("d_input", 128)
    LAYER_TYPE = _model.get("layer_type", 'Conv1D')
    N_LAYERS = _model.get("n_layers", 4)
    N_HEADS = _model.get("n_heads", 4)
    D_MODEL = _model.get("d_model", 256)
    D_FF = _model.get("d_ff", 128)
    # D_DENSE = _model.get("d_dense", 128)


""" Const """
PADDING_VAL = -10000.
CLS_VAL = 60.

""" Output """
OUTPUT_DIR = conf.get("OUTPUT_DIR", f"./results/{DATASET.FEATURE}-{DATASET.MIN_TIME_STEP}-{DATASET.MAX_TIME_STEP}_{MODEL.LAYER_TYPE}-i{MODEL.D_INPUT}-l{MODEL.N_LAYERS}-h{MODEL.N_HEADS}-f{MODEL.D_FF}-d{MODEL.D_MODEL}")
classes = {}
try:
    with open(f"{DATASET.DIR}/list.json") as f:
        for i, k in enumerate(json.load(f)['train'].keys()):
            classes[k] = i
except (FileNotFoundError, ValueError):
    logging.warning("Invalid dataset list.json!")
