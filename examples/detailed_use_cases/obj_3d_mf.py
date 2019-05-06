"""
  Synthetic function for 3D multi-fidelity optimisation.
  -- kirthevasank
"""

def objective(z, x):
  """ Computes the objectives. """
  normalised_z = (z[0] - 60.0)/ 120.0
  vol1 = x[0] # LiNO3
  vol2 = x[1] # Li2SO4
  vol3 = x[2] # NaClO4
  vol4 = 10 - (vol1 + vol2 + vol3) # Water
  # Synthetic functions
  conductivity = vol1 * normalised_z + \
                 0.1 * (vol2 + vol3) ** 2 * (normalised_z ** 1.5) + \
                 2.3 * vol4 * (vol1 ** 1.5)
  return conductivity


def cost(z):
  """ Cost function. """
  normalised_z = (z[0] - 60.0)/ 120.0
  return 0.1 + 0.9 * normalised_z 

