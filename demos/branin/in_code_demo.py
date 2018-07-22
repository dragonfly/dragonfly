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
  domain_bounds = [[-5, 10], [0, 15]]
  max_capital = 100 
  opt_val, opt_pt = maximise_function(branin, domain_bounds, max_capital,
                                      hp_tune_criterion='post_sampling',
                                      hp_tune_method='slice')
  print('Optimum Value in %d evals: %0.4f'%(max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))

if __name__ == '__main__':
  main()

