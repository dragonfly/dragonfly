"""
  Parkd function with multi-fidelity.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from .park2_4 import park2_4_z

# Write a function like this called 'obj'.
def park2_4_mf(z, x):
  """ Computes the Parkd function. """
  return park2_4_z(z[0], x)

def objective(z, x):
  """ Objective. """
  return park2_4_mf(z, x)


def cost(z):
  """ Cost function. """
  return 0.05 + 0.95 * z[0]**1.5


def main(z, x):
  """ main function. """
  return park2_4_mf(z, x), cost(z)

