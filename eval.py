import json
import tensorflow as tf
from tqdm import tqdm
import env
from dataset import get_dataset
# from model import get_model
from tweet_net import transformer_based_model as get_model
import numpy as np
import os
from sklearn.metrics import f1_score
m = get_model(n_classes=len(env.classes.keys()))
with open(f"{env.DATASET.DIR}/list.json") as f:
    ds = json.load(f)
    test_set = ds['test']

confusion_matrix = np.zeros((len(env.classes.keys()), len(env.classes.keys())), dtype=np.uint)
# (actual, preds)
ds, ts = get_dataset(test_set, env.TRAIN.BATCH_SIZE, repeat=False)
m.load_weights(f"{env.OUTPUT_DIR}/checkpoints/ckpt")

total = 0
correct = 0
trues_global = []
preds_global = []
for x, y in ds:
    preds = m.predict(x)
    preds_label = tf.math.argmax(preds, axis=-1).numpy()
    for real, pred in zip(y.numpy(), preds_label):
        total += 1
        correct += 1 if real == pred else 0
        confusion_matrix[real][pred] += 1
        trues_global.append(real)
        preds_global.append(pred)
print(f"Total   Accuracy: {(correct / total) * 100:.5f}%")
class_avg_acc = np.mean([ x[i] / np.sum(x) for i, x in enumerate(confusion_matrix)])
print(f"Average Accuracy: {class_avg_acc*100:.5f}%")
print(confusion_matrix)

os.system(f"mkdir -p {env.OUTPUT_DIR}")
with open(f"{env.OUTPUT_DIR}/metrics.txt", 'w') as f:
    f.write(f"{env.OUTPUT_DIR}/metrics.txt" + "\n")
    f1 = f1_score(trues_global, preds_global, average='macro')
    print(f1)
    f.write("  Marco F1-score: " + str(f1*100) + "%\n")
    f.write("  Total Accuracy: " + str((correct / total)*100) + "%\n")
    f.write("  Aver. Accuracy: " + str(class_avg_acc*100) + "%\n")

np.save(f"{env.OUTPUT_DIR}/confusion_matrix.npy", confusion_matrix)

