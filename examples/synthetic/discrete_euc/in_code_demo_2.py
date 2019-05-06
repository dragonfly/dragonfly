"""
  In code demo for discrete euclidean domains
"""

import numpy as np
from dragonfly import load_config, maximise_function


def hartmann6_3(x):
  """ Hartmann function in 3D. """
  pt = np.array([x[1][1]/11.0,
                 x[0][0],
                 x[0][2],
                 x[0][1],
                 x[2]/114.0,
                 x[1][0]/11.0,
                ])
  A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                [0.05,  10,   17, 0.1,   8, 14],
                [   3, 3.5,  1.7,  10,  17,  8],
                [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
  P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return alpha.dot(np.exp(-log_sum_terms))


def main():
  """ Main function. """
  disc_euc_items = list(np.random.random((1000, 3)))
  domain_vars = [{'type': 'discrete_euclidean', 'items': disc_euc_items},
                 {'type': 'float',  'min': 0, 'max': 11, 'dim': 2},
                 {'type': 'int',  'min': 0, 'max': 114},
                ]
  config_params = {'domain': domain_vars}
  config = load_config(config_params)
  max_num_evals = 100
  opt_val, opt_pt, history = maximise_function(hartmann6_3, config.domain, max_num_evals,
                                               config=config)
  print(opt_pt, opt_val)


if __name__ == '__main__':
  main()

