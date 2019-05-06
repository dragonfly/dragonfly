"""
  Prior mean functions used in the examples in this directory.
  -- kirthevasank
"""

# We are simply collecting all the prior mean functions used in the demos here for
# neatness. They need not be defined this way.

# This is the prior mean we will use for the conductivity in the 3D example.
# We use this in in_code_demo_multi_objective.py and in_code_demo_single_objective.py
def conductivity_prior_mean_3d(x):
  """ Prior mean for the conductivity. """
  vol1 = x[0] # LiNO3
  vol2 = x[1] # Li2SO4
  vol3 = x[2] # NaClO4
  vol4 = 10 - (vol1 + vol2 + vol3) # Water
  return 0.9 * vol1 + 0.12 * (vol2 + vol3) ** 2.05 + 2.4 * vol4 * (vol1 ** 1.6)


# Prior mean for the conductivity in the 3D example with multi-fidelity.
# See in_code_demo_single_objective.py
def conductivity_prior_mean_3d_mf(z, x):
  """ Prior mean for the conductivity with multi-fidelity. """
  normalised_z = (z[0] - 60.0)/ 120.0
  vol1 = x[0] # LiNO3
  vol2 = x[1] # Li2SO4
  vol3 = x[2] # NaClO4
  vol4 = 10 - (vol1 + vol2 + vol3) # Water
  # Synthetic functions
  return 0.9 * vol1 * normalised_z + \
         0.12 * (vol2 + vol3) ** 2.005 * (normalised_z ** 1.55) + \
         2.4 * vol4 * (vol1 ** 1.5)


# Prior mean for the conductivity in the 5D example.
# See in_code_demo_multi_objective.py and in_code_demo_single_objective.py
def conductivity_prior_mean_5d(x):
  """ Prior mean for the conductivity. """
  vol1 = x[0] # Na2SO4
  vol2 = x[1] # NaNO3
  vol3 = x[2] # LiNO3
  vol4 = x[3] # Li2SO4
  vol5 = x[4] # NaClO4
  vol6 = 10 - (vol1 + vol2 + vol3 + vol4 + vol5) # Water
  # Synthetic functions
  return 1.01 * (vol1 + vol5) + 0.15 * (vol2 + vol3) ** 2.05 + \
         0.76 * vol4 * vol3 + 2.2 * vol6 * (vol1 ** 1.6)

