
Project studying convergence for Game Of Life based on paper [Springer, J. & Kenyon, G. (2009)](https://arxiv.org/pdf/2009.01398)


TODO:
- saving model
- drawing loss landscapes for multiple trajectories (origin for PCA: average between last params, best params)

For parameters below 2/64 reached convergence (i = 1, 17)
```
training_data_cardinality = 2**16
training_data_seed = 1

board_shape = (32, 32)
overcompleteness = 1

batch_size = 8
epochs = 64
n_instances = 64
```

```
ds = random_uniform(0.38, (training_data_cardinality, *board_shape), training_data_seed)

```
```
optimizer = keras.optimizers.Adam(0.001)
model = build_model(board_shape, overcompleteness, i+100, optimizer)

```

