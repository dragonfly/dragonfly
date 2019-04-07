"""
  In code demo for borehole_constrained example.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function
from dragonfly import load_config, maximise_function, maximise_multifidelity_function
# From current directory
from borehole_constrained  import objective
from borehole_constrained_mf import objective as mf_objective
from borehole_constrained_mf import cost as mf_cost


def main():
  """ Main function. """
  # First Specify all parameters
  domain_vars = [{'name': 'rw', 'type': 'float', 'min': 0.05, 'max': 0.15, 'dim': 1},
                 {'name': 'L_Kw', 'type': 'float',  'min': 0, 'max': 1, 'dim': 2},
                 {'name': 'Tu', 'type': 'int',  'min': 63070, 'max': 115600, 'dim': ''},
                 {'name': 'Tl', 'type': 'float',  'min': 63.1, 'max': 116},
                 {'name': 'Hu_Hl', 'type': 'int',  'min': 0, 'max': 240, 'dim': 2},
                 {'name': 'r', 'type': 'float',  'min': 100, 'max': 50000},
                ]
  domain_constraints = [{'constraint': 'np.sqrt(rw[0]) + L_Kw[1] <= 0.9'},
                        {'constraint': 'r/100.0 + Hu_Hl[1] < 200'}
                       ]
  fidel_vars = [{'name': 'fidel_0', 'type': 'float',  'min': 0.05, 'max': 0.25},
                {'name': 'fidel_1', 'type': 'discrete_numeric', 'items': "0.1:0.05:1.01"},
               ]
  fidel_space_constraints = [
    {'name': 'fsc1', 'constraint': 'fidel_0 + fidel_1 <= 0.9'}
    ]
  fidel_to_opt = [0.1, 0.75]
  # Budget of evaluations
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  max_mf_capital = max_num_evals * mf_cost(fidel_to_opt) # Multi-fideltiy capital

  # First do the MF version
  config_params = {'domain': domain_vars, 'fidel_space': fidel_vars,
                   'domain_constraints': domain_constraints,
                   'fidel_space_constraints': fidel_space_constraints,
                   'fidel_to_opt': fidel_to_opt}
  config = load_config(config_params)
  # Optimise
  mf_opt_pt, mf_opt_val, history = maximise_multifidelity_function(mf_objective,
                                     config.fidel_space, config.domain,
                                     config.fidel_to_opt, mf_cost, 
                                     max_mf_capital, config=config)
  print(mf_opt_pt, mf_opt_val)

  # Non-MF version
  config_params = {'domain': domain_vars, 'domain_constraints': domain_constraints}
  config = load_config(config_params)
  max_capital = 100 # Optimisation budget (max number of evaluations)
  # Optimise
  opt_pt, opt_val, history = maximise_function(objective, config.domain,
                                               max_num_evals, config=config)
  print(opt_pt, opt_val)


if __name__ == '__main__':
  main()

