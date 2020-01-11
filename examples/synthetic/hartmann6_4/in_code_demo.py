"""
  In code demo for Hartmann6_4
  -- kandasamy@cs.cmu.edu
"""

from dragonfly import load_config, maximise_function, maximise_multifidelity_function
# From current directory
from hartmann6_4  import objective
from hartmann6_4_mf import objective as mf_objective
from hartmann6_4_mf import cost as mf_cost


def main():
  """ Main function. """
  # First Specify all parameters
  domain_vars = [{'type': 'int', 'min': 224, 'max': 324, 'dim': 1},
                 {'type': 'float',  'min': 0, 'max': 10, 'dim': 2},
                 {'type': 'float',  'min': 0, 'max': 1, 'dim': 1},
                 {'type': 'int',  'min': 0, 'max': 92, 'dim': 2},
                ]
  fidel_vars = [{'type': 'float',  'min': 1234.9, 'max': 9467.18, 'dim': 2},
                {'type': 'discrete',  'items': ['a', 'bc', 'def', 'ghij']},
                {'type': 'int',  'min': 123, 'max': 234, 'dim': 1},
               ]
  fidel_to_opt = [[9467.18, 9452.8], "def", [234]]
  # Budget of evaluations
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  max_mf_capital = max_num_evals * mf_cost(fidel_to_opt) # Multi-fideltiy capital

  # First do the MF version
  config_params = {'domain': domain_vars, 'fidel_space': fidel_vars,
                   'fidel_to_opt': fidel_to_opt}
  config = load_config(config_params)
  opt_method = 'bo'
#   opt_method = 'rand'
  # Optimise
  mf_opt_val, mf_opt_pt, history = maximise_multifidelity_function(mf_objective,
                                     config.fidel_space, config.domain,
                                     config.fidel_to_opt, mf_cost, 
                                     max_mf_capital, config=config,
                                     opt_method=opt_method)
  print(mf_opt_pt, mf_opt_val)

  # Non-MF version
  config_params = {'domain': domain_vars}
  config = load_config(config_params)
  max_capital = 100 # Optimisation budget (max number of evaluations)
  # Optimise
  opt_val, opt_pt, history = maximise_function(objective, config.domain,
                                               max_num_evals, config=config, opt_method=opt_method)
  print(opt_pt, opt_val)


if __name__ == '__main__':
  main()

