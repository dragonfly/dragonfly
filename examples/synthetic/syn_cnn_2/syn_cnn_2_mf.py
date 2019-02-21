"""
  A synthetic function on CNNs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

# Local
from .syn_cnn_2 import syn_cnn_2_z_x


def syn_cnn_2_mf(z, x):
  """ Computes the Branin function. """
  f = [(z[0][1]-10)/4, (z[0][0]-10)/4]
  return syn_cnn_2_z_x(f, x)


def cost(z):
  """ Cost function. """
  f = [(z[0][1]-10)/4, (z[0][0]-10)/4]
  return sum(f)/2.0


# Write a function like this called 'obj'.
def objective(z, x):
  """ Objective. """
  return syn_cnn_2_mf(z, x)


def main(z, x):
  """ main function. """
  return syn_cnn_2_mf(z, x), cost(z)

