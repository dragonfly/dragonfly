"""
  Branin function.
  -- kvysyara@andrew.cmu.edu
"""

from __future__ import print_function
import numpy as np
import math
from maximise_function import maximise_function
# Local imports
from demos.branin.branin import branin

def main():
  domain_bounds = [[0, 1], [0, 1]]
  max_capital = 25
  opt_val, opt_pt = maximise_function(branin, domain_bounds, max_capital,
                                      hp_tune_criterion='post_sampling',
                                      hp_tune_method='slice')
  print('Optimum Point: ', end='')
  print(opt_pt)
  print('Optimum Value: ', end='')
  print(opt_val)

if __name__ == '__main__':
  main()

