import numpy as np

class KeyboardEvents:
  def __init__(self):
    self.array_buf = []

  def _to_binary_number(self, boolean):
    return 1.0 if boolean else 0.0

  def use_file(self, data):
    run_length = 16

    if not isinstance(data["keyboardEvents"], list) or len(data["keyboardEvents"]) < run_length:
      # Fail-safe for non-existent or too small keyboard data
      return

    keyboard_events = data["keyboardEvents"]
    arrays = []

    for i in range(len(keyboard_events) // run_length):
      kbd_run = keyboard_events[i * run_length : (i + 1) * run_length]

      min_ts = kbd_run[0]["timestamp"]
      max_ts = kbd_run[run_length - 1]["timestamp"]

      arrays.append(np.array([[self._to_binary_number(ev["down"]), (ev["timestamp"] - min_ts) / (max_ts - min_ts)] for ev in kbd_run]))

    self.array_buf += arrays
  