import json
import logging
import os

TIME_STEP = 1800
MIN_TIME_STEP = 450
BATCH_SIZE = int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 32
if "BATCH_SIZE" in os.environ:
    logging.info(f"set batch size from {BATCH_SIZE} environment variable")

classes = {}
try:
    with open("dataset/list.json") as f:
        for i, k in enumerate(json.load(f)['train'].keys()):
            classes[k] = i
except Exception:
    logging.warning("Invalid dataset list.json!")
test_data_ratio = .2

PADDING_VAL = -10000.
CLS_VAL = 60.
