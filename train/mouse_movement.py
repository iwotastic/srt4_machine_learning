import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

mouse_movements = tf.keras.models.Sequential()
mouse_movements.add(tf.keras.layers.Conv2D(4, 4, input_shape=(600, 600, 3), activation="relu"))
mouse_movements.add(tf.keras.layers.Conv2D(6, 4, activation="relu"))
mouse_movements.add(tf.keras.layers.Flatten())
mouse_movements.add(tf.keras.layers.Dense(2, activation="relu"))
mouse_movements.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

chunks = 20
metric = "mouse"
size = 200
epoch_bundles = 10

for i in range(epoch_bundles):
  print(f"==== EPOCH BUNDLE {i + 1} ====")

  chunk_beginning = list(range(chunks))
  shuffle(chunk_beginning)
  chunk_list = chunk_beginning + [chunks]
  
  for chunk in chunk_list:
    print(f"Chunk {chunk}:" if chunk != chunks else "Validation")

    with open(f"/Volumes/SRT4Data/bot_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as bot_mouse_data_file:
      bot_mouse_data = pickle.load(bot_mouse_data_file)

    with open(f"/Volumes/SRT4Data/human_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as human_mouse_data_file:
      human_mouse_data = pickle.load(human_mouse_data_file)

    x_train = np.stack(bot_mouse_data + human_mouse_data)
    y_train = np.stack([np.array([0.0, 1.0])] * len(bot_mouse_data) + [np.array([1.0, 0.0])] * len(human_mouse_data))

    if chunk != chunks:
      mouse_movements.fit(x_train, y_train)
    else:
      print(f"Loss: {mouse_movements.evaluate(x_train, y_train)}\n")

mouse_movements.save("model/mouse_movements")