"""
  Fidelity space constraint for hartmann3_constrained example.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

def constraint(z):
  """ Evaluate the constraint here. """
  # The following is true in [1, 2) \cup [3, 4) \cup ... \cup [9, 10)
  ret1 = int(z[0]) % 2 == 1
  # The following is true in [0, 1) \cup [3, 4) \cup  [6, 7) \cup [9, 10)
  ret2 = int(z[0]) % 3 == 0
  return ret1 or ret2

