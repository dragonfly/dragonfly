"""
  Hartmann function in 3 dimensions with constraints.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

try:
  from hartmann3_constrained_mf import hartmann3_constrained_mf, OPT_TO_FIDEL_PT
except ImportError:
  from .hartmann3_constrained_mf import hartmann3_constrained_mf, OPT_TO_FIDEL_PT


def hartmann3_constrained(x):
  """ Exactly the same as the Hartmann3 function. """
  return hartmann3_constrained_mf([OPT_TO_FIDEL_PT], x)


# Write a function like this called objective.
def objective(x):
  """ Objective. """
  return hartmann3_constrained(x)

