"""
  Parkd2 function.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

try:
  from .park2_3 import park2_3_z_x
except ImportError:
  from park2_3 import park2_3_z_x


def park2_3_mf(z, x):
  """ Computes the Park2_3 function. """
  f = [z[0][0]/5000.0, z[1]/10.0]
  return park2_3_z_x(f, x)

# Write a function like this called 'obj'....
def cost(z):
  """ Cost function. """
  return 0.1 + 0.9 * (z[0][0]/5000 + z[1]/10) / 2.0


def objective(z, x):
  """ Objective. """
  return park2_3_mf(z, x)


def main(z, x):
  """ main function. """
  return park2_3_mf(z, x), cost(z)

