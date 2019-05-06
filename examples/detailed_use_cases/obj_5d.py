"""
  Synthetic function for 5D optimisation.
  -- kirthevasank
"""

from moo_5d import compute_objectives as moo_objectives

def objective(x):
  """ Computes the objectives. """
  return moo_objectives(x)[0] # Just returns conductivity

