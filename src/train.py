from model import build_model
from dataset import random_uniform

import tensorflow as tf
import keras

losses = []

if __name__ == "__main__":
    board_shape = (16, 16)
    ds = random_uniform(0.38, (32766, *board_shape), 0)
    ds = ds.batch(16)
    for i in range(64):
        optimizer = keras.optimizers.SGD(0.01)
        model = build_model(board_shape, 1, i, optimizer)
        # model.summary()

        early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=1, verbose=1)
        history = model.fit(
            ds,
            epochs = 32,
            callbacks = [early_stopping],
            verbose = 1
        )
        loss = min(history.history["loss"])
        print(i, "done with loss", loss)
        losses.append(loss)
