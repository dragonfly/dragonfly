"""
  Multi-fidelity Branin function.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from demos_synthetic.branin.branin import branin_z_x


def branin_mf(z, x):
  """ Multi-fidelity Branin function. """
  return branin_z_x([z[0], z[1][0], z[1][1]], x)


def obj(z, x):
  """ Objective. """
  return branin_mf(z, x)


def cost(z):
  """ Cost function. """
  return 0.05 + 0.95 * z[0]**1.5


def main(z, x):
  """ main function. """
  return obj(z, x), cost(z)

