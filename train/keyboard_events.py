import tensorflow as tf
import numpy as np
import pickle
from random import shuffle

keyboard_events = tf.keras.models.Sequential()
keyboard_events.add(tf.keras.layers.SimpleRNN(8, activation="relu"))
keyboard_events.add(tf.keras.layers.Dense(2))
keyboard_events.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

chunks = 40
metric = "kbd"
size = 200
epoch_bundles = 10
training_chunks = [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_1.dat"
) for chunk_num in range(40)] + [(
  f"preprocessed_data/human_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat",
  f"preprocessed_data/bot_{metric}_chunk_{chunk_num}_size_{size}_round_2.dat"
) for chunk_num in range(13)]

validation_chunk = (
  f"preprocessed_data/human_{metric}_chunk_40_size_{size}_round_1.dat",
  f"preprocessed_data/bot_{metric}_chunk_40_size_{size}_round_1.dat"
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
      keyboard_events.fit(x_train, y_train)
    else:
      print(f"Loss: {keyboard_events.evaluate(x_train, y_train)}\n")

keyboard_events.save("model/keyboard_events_v2")