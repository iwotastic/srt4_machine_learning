import numpy as np

class BrowserAttributes:
  def __init__(self):
    self.array_buf = []

  def _to_binary_number(self, boolean):
    return 1.0 if boolean else 0.0

  def use_file(self, data):
    self.array_buf.append(np.array([
      self._to_binary_number(data["oneTimeChecks"]["mouseEventsPossible"]),
      self._to_binary_number(data["oneTimeChecks"]["touchEventsPossible"]),
      self._to_binary_number(data["oneTimeChecks"]["keyEventsPossible"]),
      self._to_binary_number(data["oneTimeChecks"]["lightMode"]),
      self._to_binary_number(data["oneTimeChecks"]["isSRGB"]),
      self._to_binary_number(data["oneTimeChecks"]["hasFinePointer"]),
      self._to_binary_number(data["oneTimeChecks"]["hasVibration"]),
      self._to_binary_number(data["oneTimeChecks"]["notReducedMotion"]),
      self._to_binary_number(data["oneTimeChecks"]["hasWebGL"]),
      self._to_binary_number(data["oneTimeChecks"]["hasWebGL2"]),
      self._to_binary_number(data["oneTimeChecks"]["hasGamepads"]),
    ]))
  