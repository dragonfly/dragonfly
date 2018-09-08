"""
  Branin function.
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import numpy as np

def branin(x):
  """ Computes the Branin function. """
  x1 = x[0]
  x2 = x[1]
  a = 1
  b = 5.1/(4*np.pi**2)
  c = 5/np.pi
  r = 6
  s = 10
  t = 1/(8*np.pi)
  neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
  return -neg_ret


# Write a function like this called 'main'
def main(x):
  """ main function. """
  return branin(x)

