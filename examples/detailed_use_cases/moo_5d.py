"""
  Synthetic multi-objective function for 3D optimisation.
  -- kirthevasank
"""

import numpy as np

num_objectives = 2

def compute_objectives(x):
  """ Computes the objectives. """
  vol1 = x[0] # Na2SO4
  vol2 = x[1] # NaNO3
  vol3 = x[2] # LiNO3
  vol4 = x[3] # Li2SO4
  vol5 = x[4] # NaClO4
  vol6 = 10 - (vol1 + vol2 + vol3 + vol4 + vol5) # Water
  # Synthetic functions
  conductivity = 0.9 * (vol1 + vol5) + 0.1 * (vol2 + vol3) ** 2 + \
                 0.8 * vol4 * vol3 + 2.1 * vol6 * (vol1 ** 1.5)
  voltage_window = 0.5 * (vol3 + vol2) ** 1.7 + 0.9 * vol1 * vol4 + \
                   1.2 * (vol4 ** 0.5) * (vol6 ** 1.5)
  # Add Gaussian noise to simulate experimental noise
  conductivity += np.random.normal() * 0.01
  voltage_window += np.random.normal() * 0.01
  return [conductivity, voltage_window]

