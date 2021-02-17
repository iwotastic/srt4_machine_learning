from pathlib import Path
import pickle
import json
from random import sample

from mouse_movement import MouseMovements # type: ignore
from browser_attributes import BrowserAttributes # type: ignore
from keyboard_events import KeyboardEvents # type: ignore

exporters = [
  (KeyboardEvents, "kbd", 200),
  (BrowserAttributes, "browser", 200),
  (MouseMovements, "mouse", 200),
]

for Exporter, name, chuck_size in exporters:
  # Set up exporter
  human_exp = Exporter()

  # Use exporter on each file
  for data_path in Path("data/human").iterdir():
    with data_path.open() as data_file:
      data = json.loads(data_file.read()[2:-1])
      human_exp.use_file(data)

  # Create random sample of indicies
  indicies = sample(range(len(human_exp.array_buf)), len(human_exp.array_buf))

  # Dump chunks
  for chunk in range(len(indicies) // chuck_size):
    with open(f"/Volumes/SRT4Data/human_{name}_chunk_{chunk}_size_{chuck_size}.dat", "wb") as save_file:
      pickle.dump(
        [human_exp.array_buf[i] for i in indicies[chunk * chuck_size : (chunk + 1) * chuck_size]],
        save_file
      )

  human_exp = None

  # Set up exporter
  bot_exp = Exporter()

  # Use exporter on each file
  for data_path in Path("data/bot").iterdir():
    try:
      with data_path.open() as data_file:
        data = json.loads(data_file.read()[2:-1])
        bot_exp.use_file(data)
    except Exception as e:
      continue

  # Create random sample of indicies
  indicies = sample(range(len(bot_exp.array_buf)), len(bot_exp.array_buf))

  # Dump chunks
  for chunk in range(len(indicies) // chuck_size):
    with open(f"/Volumes/SRT4Data/bot_{name}_chunk_{chunk}_size_{chuck_size}.dat", "wb") as save_file:
      pickle.dump(
        [bot_exp.array_buf[i] for i in indicies[chunk * chuck_size : (chunk + 1) * chuck_size]],
        save_file
      )

  bot_exp = None