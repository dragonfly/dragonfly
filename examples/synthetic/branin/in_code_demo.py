"""
  Branin function.
  -- kvysyara@andrew.cmu.edu
"""

# pylint: disable=invalid-name

from __future__ import print_function
from dragonfly import maximise_function
# Local imports
try:
  from .branin import branin
except:
  from branin import branin

def main():
  """ Main function. """
  domain_bounds = [[-5, 10], [0, 15]]
  max_capital = 100
  opt_val, opt_pt, _ = maximise_function(branin, domain_bounds, max_capital)
  print('Optimum Value in %d evals: %0.4f'%(max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))

if __name__ == '__main__':
  main()

