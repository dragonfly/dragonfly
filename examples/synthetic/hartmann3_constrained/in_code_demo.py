"""
  In code demo for hartmann3_constrained example.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function
from dragonfly import load_config, maximise_function, maximise_multifidelity_function
# From current directory
from hartmann3_constrained  import objective
from hartmann3_constrained_mf import objective as mf_objective
from hartmann3_constrained_mf import cost as mf_cost
from fsc1_constraint import constraint as fsc1_constraint


def main():
  """ Main function. """
  # First Specify all parameters
  domain_vars = [{'name': 'x', 'type': 'float', 'min': 0, 'max': 1, 'dim': 3}]
  domain_constraints = [
    {'name': 'quadrant', 'constraint': 'np.linalg.norm(x[0:2]) <= 0.5'},
    ]
  fidel_vars = [{'name': 'z', 'type': 'float',  'min': 0, 'max': 10}]
  fidel_space_constraints = [{'name': 'fsc1', 'constraint': fsc1_constraint}]
  fidel_to_opt = [9.1]
  print('fsc1_constraint(fidel_to_opt)', fsc1_constraint(fidel_to_opt))
  # Budget of evaluations
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  max_mf_capital = max_num_evals * mf_cost(fidel_to_opt) # Multi-fideltiy capital

  # Non-MF version
  config_params = {'domain': domain_vars, 'domain_constraints': domain_constraints}
  config = load_config(config_params)
  max_capital = 100 # Optimisation budget (max number of evaluations)
  # Optimise
  opt_val, opt_pt, history = maximise_function(objective, config.domain,
                                               max_num_evals, config=config)
  print(opt_pt, opt_val)

  # MF version
  config_params = {'domain': domain_vars, 'fidel_space': fidel_vars,
                   'domain_constraints': domain_constraints,
                   'fidel_space_constraints': fidel_space_constraints,
                   'fidel_to_opt': fidel_to_opt}
  config = load_config(config_params)
  # Optimise
  mf_opt_val, mf_opt_pt, history = maximise_multifidelity_function(mf_objective,
                                     config.fidel_space, config.domain,
                                     config.fidel_to_opt, mf_cost, 
                                     max_mf_capital, config=config)
  print(mf_opt_pt, mf_opt_val)


if __name__ == '__main__':
  main()

