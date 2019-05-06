"""
  In code demo for supernova experiment.
  -- kandasamy@cs.cmu.edu
"""

from dragonfly import load_config, maximise_function, maximise_multifidelity_function
# From current directory
from snls import objective as snls_objective
from snls_mf import objective as snls_mf_objective
from snls_mf import cost as snls_mf_cost


def main():
  """ Main function. """
  domain_vars = [{'name': 'hubble_constant', 'type': 'float', 'min': 60, 'max': 80},
                 {'name': 'omega_m', 'type': 'float',  'min': 0, 'max': 1},
                 {'name': 'omega_l', 'type': 'float',  'min': 0, 'max': 1}]
  fidel_vars = [{'name': 'log10_resolution', 'type': 'float', 'min': 2, 'max': 5},
                {'name': 'num_obs_to_use', 'type': 'int',  'min': 50, 'max': 192}]
  fidel_to_opt = [5, 192]
  max_capital = 2 * 60 * 60 # Optimisation budget in seconds

  # A parallel set up where we will evaluate the function in three different threads.
  num_workers = 3

  # Optimise without multi-fidelity
  config_params = {'domain': domain_vars}
  config = load_config(config_params)
  opt_val, opt_pt, history = maximise_function(snls_objective, config.domain,
                                               max_capital, num_workers=num_workers,
                                               capital_type='realtime', config=config)
  print(opt_pt, opt_val)

  # Optimise with multi-fidelity
  config_params = {'domain': domain_vars, 'fidel_space': fidel_vars,
                   'fidel_to_opt': fidel_to_opt}
  config = load_config(config_params)
  # Optimise
  mf_opt_val, mf_opt_pt, history = maximise_multifidelity_function(snls_mf_objective,
                                     config.fidel_space, config.domain,
                                     config.fidel_to_opt, snls_mf_cost,
                                     max_capital, config=config)
  print(mf_opt_pt, mf_opt_val)



if __name__ == '__main__':
  main()

