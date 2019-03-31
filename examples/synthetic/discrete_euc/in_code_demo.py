"""
  In code demo for discrete euclidean domains
"""
import numpy as np

from dragonfly import load_config, maximise_function


def objective(x):
  print(x)
  return np.linalg.norm(x)


def main():
  size = 10
  dim = 3
  space = np.random.rand(size, dim)
  domain_vars = [
      {'type': 'discrete_euclidean', 'items': space},
  ]
  config_params = {'domain': domain_vars}
  config = load_config(config_params)
  max_num_evals = 100
  opt_pt, opt_val, history = maximise_function(objective, config.domain, max_num_evals, config=config)
  print(opt_pt, opt_val)


if __name__ == '__main__':
  main()

