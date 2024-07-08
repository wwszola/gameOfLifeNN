from model import build_model
from dataset import random_uniform

import tensorflow as tf
import keras

if __name__ == "__main__":
    board_shape = (16, 16)
    ds = random_uniform(0.1, (65532, *board_shape), 64)
    model = build_model(board_shape, 10)
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=1, verbose=1)
    history = model.fit(
        ds,
        epochs = 100,
        callbacks = [early_stopping]
    )
