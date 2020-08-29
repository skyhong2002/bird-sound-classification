import numpy as np

import env
from model import get_model

m = get_model(10, True)
res, att = m(np.zeros((1, env.MAX_TIME_STEP, 128)))
print([x.shape for x in att])
