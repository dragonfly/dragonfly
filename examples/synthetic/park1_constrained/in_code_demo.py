"""
  In code demo for borehole_constrained example.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function
from dragonfly import load_config, maximise_function, maximise_multifidelity_function
# From current directory
from park1_constrained  import objective
from park1_constrained_mf import objective as mf_objective
from park1_constrained_mf import cost as mf_cost
from dc2_constraint import constraint as dc2_constraint


def main():
  """ Main function. """
  # First Specify all parameters
  disc_num_items_x1 = [4, 10, 23, 45, 78, 87.1, 91.8, 99, 75.7, 28.1, 3.141593]
  domain_vars = [
    {'name': 'x0', 'type': 'discrete_numeric', 'items': '0:0.05:1', 'dim': 2},
    {'name': 'x1', 'type': 'discrete_numeric', 'items': disc_num_items_x1},
    {'name': 'x2', 'type': 'float',  'min': 10, 'max': 16},
    ]
  domain_constraints = [
    {'name': 'dc1', 'constraint': 'sum(x0) + (x2 - 10)/6.0 <= 2.1'},
    {'name': 'dc2', 'constraint': dc2_constraint}
    ]
  disc_items_z2 = ['a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef', 'abcdefg', 'abcdefg',
                   'abcdefgh', 'abcdefghi']
  fidel_vars = [{'name': 'z1', 'type': 'discrete', 'items': disc_items_z2, 'dim': 2},
                {'name': 'z2', 'type': 'float', 'min': 21.3, 'max': 243.9},
               ]
  fidel_to_opt = [["abcdefghi", "abcdefghi"], 200.1]
  # Budget of evaluations
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  max_mf_capital = max_num_evals * mf_cost(fidel_to_opt) # Multi-fideltiy capital

  # First do the MF version
  config_params = {'domain': domain_vars, 'fidel_space': fidel_vars,
                   'domain_constraints': domain_constraints,
                   'fidel_to_opt': fidel_to_opt}
  config = load_config(config_params)
  # Optimise
  mf_opt_val, mf_opt_pt, history = maximise_multifidelity_function(mf_objective,
                                     config.fidel_space, config.domain,
                                     config.fidel_to_opt, mf_cost, 
                                     max_mf_capital, config=config)
  print(mf_opt_pt, mf_opt_val)

  # Non-MF version
  config_params = {'domain': domain_vars, 'domain_constraints': domain_constraints}
  config = load_config(config_params)
  max_capital = 100 # Optimisation budget (max number of evaluations)
  # Optimise
  opt_val, opt_pt, history = maximise_function(objective, config.domain,
                                               max_num_evals, config=config)
  print(opt_pt, opt_val)


if __name__ == '__main__':
  main()

