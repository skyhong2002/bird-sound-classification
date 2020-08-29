import json
import os

import env
from dataset import get_dataset
from model import get_model
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

N_ITEMS = int(os.environ['N_ITEMS']) if 'N_ITEMS' in os.environ else 100

m = get_model(n_classes=len(env.classes.keys()))

with open("dataset/list.json") as f:
    ds = json.load(f)
    test_set = ds['test']

confusion_matrix = np.zeros((len(env.classes.keys()), len(env.classes.keys())), dtype=np.uint)
# (actual, preds)
ds, ts = get_dataset(test_set, env.BATCH_SIZE, repeat=False)
m.load_weights("checkpoints/ckpt")
m2 = Model(inputs=m.inputs, outputs=m.layers[-2].output)
m2.load_weights("checkpoints/ckpt")
embeddings = []
labels = []

cnt = 0
for x, y in ds:
    cnt += y.shape[0]
    preds = m2.predict(x)
    for i in range(preds.shape[0]):
        embeddings.append(preds[i])
        labels.append(y[i])
    if cnt > N_ITEMS:
        break

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
x_tsne = tsne.fit_transform(embeddings)
keys = list(env.classes.keys())
data = [[] for _ in keys]
for i, (px, py) in enumerate(x_tsne):
    data[labels[i]].append((px, py))

for i, l in enumerate(data):
    try:
        plt.scatter([(x, _) for (x, _) in l], [(y, _) for (_, y) in l], label=keys[i])
    except Exception as e:
        print(e)

print(f"{cnt} items.")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig("t_sne.png")
plt.show()
