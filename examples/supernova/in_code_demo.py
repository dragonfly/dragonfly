"""
  In code demo for supernova experiment.
  -- kandasamy@cs.cmu.edu
"""

from argparse import Namespace
from dragonfly import load_config, maximise_function
# From current directory
from snls import objective


def main():
  """ Main function. """
  domain_vars = [{'name': 'hubble_constant', 'type': 'float', 'min': 60, 'max': 80},
                 {'name': 'omega_m', 'type': 'float',  'min': 0, 'max': 1},
                 {'name': 'omega_l', 'type': 'float',  'min': 0, 'max': 1}]
  config_params = {'domain': domain_vars}
  config = load_config(config_params)
  objective = 
  max_capital = 2 * 60 * 60 # Optimisation budget in seconds

  # Optimise
  opt_pt, opt_val, history = maximise_function(objective, config.domain,
                               max_capital, capital_type='realtime', config=config)


if __name__ == '__main__':
  main()

