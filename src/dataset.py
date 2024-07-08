from conway import next_state

import tensorflow as tf
import numpy as np

def random_uniform(density, shape, batch_size = 8):
    rng = np.random.default_rng()
    prev_states = rng.choice(2, shape, p = [1.0-density, density]).astype(np.uint8)
    next_states = np.array([next_state(state) for state in prev_states])
    ds = tf.data.Dataset.from_tensor_slices((prev_states, next_states))
    ds = ds.batch(batch_size)
    return ds

