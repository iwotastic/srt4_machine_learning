import numpy as np
import tensorflow as tf

class MouseMovements:
  def __init__(self):
    self.array_buf = []

  def use_file(self, data):
    if not isinstance(data["mouseMovements"], list) or len(data["mouseMovements"]) == 0:
      # Fail-safe for non-existent mouse movement data
      return

    no_clicks = ("mouseClicks" not in data) or (len(data["mouseClicks"]) == 0)

    split_mouse_movements = []
    movement_buffer = []
    movement_cursor = 0
    click_cursor = 0

    while movement_cursor < len(data["mouseMovements"]):
      movement = data["mouseMovements"][movement_cursor]

      if no_clicks:
        movement_buffer.append((movement["x"], movement["y"], movement["timestamp"]))
      elif click_cursor >= len(data["mouseClicks"]):
        movement_buffer.append((movement["x"], movement["y"], movement["timestamp"]))
      elif movement["timestamp"] < data["mouseClicks"][click_cursor]["timestamp"]:
        movement_buffer.append((movement["x"], movement["y"], movement["timestamp"]))
      elif movement["timestamp"] > data["mouseClicks"][click_cursor]["timestamp"]:
        click = data["mouseClicks"][click_cursor]
        movement_buffer.append((click["x"], click["y"], click["timestamp"]))
        split_mouse_movements.append(movement_buffer)
        movement_buffer = []
        click_cursor += 1
        movement_buffer.append((movement["x"], movement["y"], movement["timestamp"]))

      movement_cursor += 1

    if click_cursor < len(data["mouseClicks"]):
      click = data["mouseClicks"][click_cursor]
      movement_buffer.append((click["x"], click["y"], click["timestamp"]))

    split_mouse_movements.append(movement_buffer)

    arrays = []

    for mouse_movement in split_mouse_movements:
      array = np.zeros((600, 600, 3))

      xs = [x for x, _, _ in mouse_movement]
      ys = [y for _, y, _ in mouse_movement]
      timestamps = [timestamp for _, _, timestamp in mouse_movement]

      # Set up key values
      min_x = min(xs)
      max_x = max(xs) if max(xs) != min_x else min_x + 1
      min_y = min(ys)
      max_y = max(ys) if max(ys) != min_y else min_y + 1
      min_timestamp = min(timestamps)
      max_timestamp = max(timestamps) if max(timestamps) != min_timestamp else min_timestamp + 1

      # Set up normalizers
      normalize_x = lambda x: int(((x - min_x) / (max_x - min_x)) * 600)
      normalize_y = lambda y: int(((y - min_y) / (max_y - min_y)) * 600)
      normalize_timestamp = lambda timestamp: (timestamp - min_timestamp) / (max_timestamp - min_timestamp)

      for index, point in enumerate(mouse_movement):
        nx = normalize_x(point[0])
        ny = normalize_y(point[1])

        # Red (index-based scaling)
        array[(nx - 3 if nx - 3 > -1 else 0) : (nx + 3 if nx + 3 < 600 else 600),
          (ny - 3 if ny - 3 > -1 else 0) : (ny + 3 if ny + 3 < 600 else 600), 0] = index / len(mouse_movement)

        # Green (Time-based scaling)
        array[(nx - 3 if nx - 3 > -1 else 0) : (nx + 3 if nx + 3 < 600 else 600),
          (ny - 3 if ny - 3 > -1 else 0) : (ny + 3 if ny + 3 < 600 else 600), 1] = normalize_timestamp(point[2])

        # Blue (Statically valued indicator of mouse path)
        array[(nx - 3 if nx - 3 > -1 else 0) : (nx + 3 if nx + 3 < 600 else 600),
          (ny - 3 if ny - 3 > -1 else 0) : (ny + 3 if ny + 3 < 600 else 600), 2] = 0.5

      # Convert to SparseTensor for storage efficiency
      arrays.append(tf.sparse.from_dense(tf.convert_to_tensor(array)))

    self.array_buf += arrays
  