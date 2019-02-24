"""
  Park1 function with three domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
from .park1_3 import park1_3_z_x


def park1_3_mf(z, x):
  """ Computes the park1 function. """
  f0 = len(z[0][0])/10.0
  f1 = len(z[0][1])/10.0
  f2 = (z[1] - 21.3) / (243.9 - 21.3)
  f = [f0, f1, f2]
  return park1_3_z_x(f, x)


def cost(z):
  """ Cost function. """
  f0 = len(z[0][0])/10.0
  f1 = len(z[0][1])/10.0
  f2 = (z[1] - 21.3) / (243.9 - 21.3)
  return 0.1 + (f0 + f1)/2 + 1.3 * f1 * f2**2


# Write a function like this called obj.
def objective(z, x):
  """ Objective. """
  return park1_3_mf(z, x)


def main(z, x):
  """ main function. """
  return park1_3_mf(z, x), cost(z)

