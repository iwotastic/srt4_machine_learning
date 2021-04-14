import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

browser_attributes = tf.keras.models.Sequential()
browser_attributes.add(tf.keras.Input(shape=(11,)))
browser_attributes.add(tf.keras.layers.Dense(32, activation="relu"))
browser_attributes.add(tf.keras.layers.Dense(2))
browser_attributes.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

chunks = 2
metric = "browser"
size = 200
epoch_bundles = 10
training_chunks = [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat"
) for chunk_num in range(2)] + [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat"
) for chunk_num in range(1)]

validation_chunk = (
  f"preprocessed_data/human_{metric}_chunk_2_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_2_size_{size}_round_1.dat"
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

    x_train = np.stack(bot_mouse_data + human_mouse_data)
    y_train = np.stack([np.array([0.0, 1.0])] * len(bot_mouse_data) + [np.array([1.0, 0.0])] * len(human_mouse_data))

    if (human_chunk, bot_chunk) != validation_chunk:
      browser_attributes.fit(x_train, y_train)
    else:
      print(f"Loss: {browser_attributes.evaluate(x_train, y_train)}\n")

browser_attributes.save("model/browser_attributes_v2")