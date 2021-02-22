import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

browser_attributes = tf.keras.models.Sequential()
browser_attributes.add(tf.keras.Input(shape=(11,)))
browser_attributes.add(tf.keras.layers.Dense(32, activation="relu"))
browser_attributes.add(tf.keras.layers.Dense(2))
browser_attributes.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy())

chunks = 2
metric = "browser"
size = 200
epoch_bundles = 10

for i in range(epoch_bundles):
  print(f"==== EPOCH BUNDLE {i + 1} ====")

  chunk_beginning = list(range(chunks))
  shuffle(chunk_beginning)
  chunk_list = chunk_beginning + [chunks]
  
  for chunk in chunk_list:
    print(f"Chunk {chunk}:" if chunk != chunks else "Validation")

    with open(f"/Volumes/SRT4Data/bot_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as bot_browser_data_file:
      bot_browser_data = pickle.load(bot_browser_data_file)

    with open(f"/Volumes/SRT4Data/human_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as human_browser_data_file:
      human_browser_data = pickle.load(human_browser_data_file)

    x_train = np.vstack(bot_browser_data + human_browser_data)
    y_train = np.vstack([np.array([0.0, 1.0])] * len(bot_browser_data) + [np.array([1.0, 0.0])] * len(human_browser_data))

    if chunk != chunks:
      browser_attributes.fit(x_train, y_train)
    else:
      print(f"Loss: {browser_attributes.evaluate(x_train, y_train)}\n")

browser_attributes.save("model/browser_attributes")