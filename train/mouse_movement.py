import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

mouse_movements = tf.keras.models.Sequential()
mouse_movements.add(tf.keras.layers.Conv2D(4, 4, input_shape=(600, 600, 3), activation="tanh"))
mouse_movements.add(tf.keras.layers.Conv2D(3, 4, activation="relu"))
mouse_movements.add(tf.keras.layers.Conv2D(2, 4, activation="tanh"))
mouse_movements.add(tf.keras.layers.Conv2D(1, 4, activation="relu"))
mouse_movements.add(tf.keras.layers.Flatten())
mouse_movements.add(tf.keras.layers.Dense(2, activation="tanh"))
mouse_movements.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

metric = "mouse"
size = 200
epoch_bundles = 10

training_chunks = [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat"
) for chunk_num in range(20)] + [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat"
) for chunk_num in range(4)]

validation_chunk = (
  f"preprocessed_data/human_{metric}_chunk_20_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_20_size_{size}_round_1.dat"
)

for i in range(epoch_bundles):
  print(f"==== EPOCH BUNDLE {i + 1} ====")

  shuffle(training_chunks)
  chunk_list = training_chunks + [validation_chunk]
  
  for chunk_num, (human_chunk, bot_chunk) in enumerate(chunk_list):
    print(f"Chunk {chunk_num}:" if (human_chunk, bot_chunk) != validation_chunk else "Validation")

    with open(bot_chunk, "rb") as bot_mouse_data_file:
      bot_mouse_data = pickle.load(bot_mouse_data_file)

    with open(human_chunk, "rb") as human_mouse_data_file:
      human_mouse_data = pickle.load(human_mouse_data_file)

    x_train = np.stack([tf.sparse.to_dense(bot_mouse_data_point).numpy() for bot_mouse_data_point in bot_mouse_data] + [tf.sparse.to_dense(human_mouse_data_point).numpy() for human_mouse_data_point in human_mouse_data])
    y_train = np.stack([np.array([0.0, 1.0])] * len(bot_mouse_data) + [np.array([1.0, 0.0])] * len(human_mouse_data))

    if (human_chunk, bot_chunk) != validation_chunk:
      mouse_movements.fit(x_train, y_train)
    else:
      print(f"Loss: {mouse_movements.evaluate(x_train, y_train)}\n")

mouse_movements.save("model/mouse_movements_v2")