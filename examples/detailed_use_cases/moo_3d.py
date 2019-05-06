"""
  Synthetic multi-objective function for 3D optimisation.
  -- kirthevasank
"""

import numpy as np

num_objectives = 2

def compute_objectives(x):
  """ Computes the objectives. """
  vol1 = x[0] # LiNO3
  vol2 = x[1] # Li2SO4
  vol3 = x[2] # NaClO4
  vol4 = 10 - (vol1 + vol2 + vol3) # Water
  # Synthetic functions
  conductivity = vol1 + 0.1 * (vol2 + vol3) ** 2 + 2.3 * vol4 * (vol1 ** 1.5)
  voltage_window = 0.5 * (vol1 + vol2) ** 1.7 + 1.2 * (vol3 ** 0.5) * (vol1 ** 1.5)
  # Add Gaussian noise to simulate experimental noise
  conductivity += np.random.normal() * 0.01
  voltage_window += np.random.normal() * 0.01
  return [conductivity, voltage_window]

