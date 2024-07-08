import tensorflow as tf
import keras

def build_model(board_shape, overcompleteness = 1):
    model = keras.Sequential()
    model.add(keras.layers.Input((*board_shape, 1)))
    for _ in range(overcompleteness):
        model.add(keras.layers.Conv2D(2, (3, 3), padding = "same", activation = "relu"))
        model.add(keras.layers.Conv2D(1, (1, 1), padding = "same", activation = "relu"))
    model.add(keras.layers.Conv2D(1, (1, 1), padding = "same", activation = "sigmoid"))

    model.compile(
        optimizer = keras.optimizers.Adam(0.01, 0.9, 0.999),
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
    )
    return model

