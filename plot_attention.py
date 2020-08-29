import env
import os

import model
from model import get_model
import numpy as np
import matplotlib.pyplot as plt

m = get_model(n_classes=len(env.classes.keys()), return_attentions=True)
m.load_weights("checkpoints/ckpt")
test_data_path = os.environ['test_data'] if 'test_data' in os.environ else \
    input("Please input test data path: ")

for path in test_data_path.split(":"):
    data_0 = np.load(path)[np.newaxis, ...]
    data = np.ones([1, env.MAX_TIME_STEP, 128]) * env.PADDING_VAL
    data[:, :data_0.shape[1], :] = data_0
    _, attentions, pad = m.predict(data)

    # a.npy (n_layers, batch, n_heads, ts, ts)
    for i, attention in enumerate(attentions):
        attention = attention[0, ...]
        fig, axes = plt.subplots(model.N_HEADS, 1, figsize=(12, 18))
        axes_r = axes.ravel()
        fig.suptitle(f"Layer {i}")
        for j in range(attention.shape[0]):
            # plt.subplot(1, 1, i + 1)
            axes_r[j].set_title(f"Head {j}")
            axes_r[j].imshow(data_0[0].T, cmap='cool', origin='lower')
            # plt.imshow([attention[j, :, 0]] * 128)
            axes_r[j].imshow([attention[j, 0, 1:data_0.shape[1]]] * 128, alpha=.5, cmap='Greys_r', origin='lower')
        fig.tight_layout()
        plt.savefig(f"attention_L{i}.png")
        # plt.show()
        plt.clf()
