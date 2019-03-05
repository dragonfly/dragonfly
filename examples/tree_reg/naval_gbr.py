"""
  Tuning the hyperparameters of Gradient boosted classification on the Protein structure
  prediction data.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from naval_gbr_mf import MAX_TR_DATA_SIZE
from naval_gbr_mf import objective as objective_mf

def objective(x):
  """ Objective. """
  return objective_mf([MAX_TR_DATA_SIZE], x)

