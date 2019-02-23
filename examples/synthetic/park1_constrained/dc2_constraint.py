"""
  Constraint for park1_constrained example.
  -- kandasamy@cs.cmu.edu
"""

def constraint(x):
  """ Evaluate the constraint here. """
  ret1 = x[2] >= 13.0 and x[1] > 11.0
  ret2 = x[2] <= 13.0 and x[1] < 90.0
  return ret1 or ret2

