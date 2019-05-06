"""
  Synthetic function for 3D optimisation.
  -- kirthevasank
"""

from moo_3d import compute_objectives as moo_objectives

def objective(x):
  """ Computes the objectives. """
  return moo_objectives(x)[0] # Just returns conductivity

