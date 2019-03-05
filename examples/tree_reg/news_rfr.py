"""
  Tuning the hyperparameters of Random forest regression on the News Popularity dataset
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from news_rfr_mf import MAX_TR_DATA_SIZE
from news_rfr_mf import objective as objective_mf

def objective(x):
  """ Objective. """
  return objective_mf([MAX_TR_DATA_SIZE], x)

