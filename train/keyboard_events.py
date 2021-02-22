import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

keyboard_events = tf.keras.models.Sequential()
keyboard_events.add(tf.keras.layers.SimpleRNN(8, activation="relu"))
keyboard_events.add(tf.keras.layers.Dense(2))
keyboard_events.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy())

chunks = 40
metric = "kbd"
size = 200
epoch_bundles = 10

for i in range(epoch_bundles):
  print(f"==== EPOCH BUNDLE {i + 1} ====")

  chunk_beginning = list(range(chunks))
  shuffle(chunk_beginning)
  chunk_list = chunk_beginning + [chunks]
  
  for chunk in chunk_list:
    print(f"Chunk {chunk}:" if chunk != chunks else "Validation")

    with open(f"/Volumes/SRT4Data/bot_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as bot_keyboard_data_file:
      bot_keyboard_data = pickle.load(bot_keyboard_data_file)

    with open(f"/Volumes/SRT4Data/human_{metric}_chunk_{chunk}_size_{size}.dat", "rb") as human_keyboard_data_file:
      human_keyboard_data = pickle.load(human_keyboard_data_file)

    x_train = np.stack(bot_keyboard_data + human_keyboard_data)
    y_train = np.stack([np.array([0.0, 1.0])] * len(bot_keyboard_data) + [np.array([1.0, 0.0])] * len(human_keyboard_data))

    if chunk != chunks:
      keyboard_events.fit(x_train, y_train)
    else:
      print(f"Loss: {keyboard_events.evaluate(x_train, y_train)}\n")

keyboard_events.save("model/keyboard_events")