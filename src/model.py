import tensorflow as tf
import keras

def build_model(board_shape, overcompleteness = 1, seed = None, optimizer = "Adam"):
    seed_gen = keras.random.SeedGenerator(seed)
    model = keras.Sequential()
    model.add(keras.layers.Input((*board_shape, 1)))
    model.add(keras.layers.Conv2D(
        filters = 2 * overcompleteness, 
        kernel_size = (3, 3), 
        padding = "same", 
        activation = "relu", 
        kernel_initializer = keras.initializers.RandomNormal(0., 1., seed_gen)
    ))
    model.add(keras.layers.Conv2D(
        filters = overcompleteness, 
        kernel_size = (1, 1), 
        padding = "same", 
        activation = "relu", 
        kernel_initializer = keras.initializers.RandomUniform(0., 1., seed_gen)
    ))
    model.add(keras.layers.Conv2D(
        filters = 1, 
        kernel_size = (1, 1), 
        padding = "same", 
        activation = "sigmoid",
        kernel_initializer = keras.initializers.RandomUniform(0., 1., seed_gen)
    ))

    model.compile(
        optimizer = optimizer,
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
    )
    return model

