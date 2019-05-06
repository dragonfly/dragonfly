"""
  In code demo for multiobjective_branin_currinexp.py
  -- kandasamy@cs.cmu.edu
"""

from dragonfly import load_config, multiobjective_maximise_functions
# From current directory
from multiobjective_branin_currinexp  import compute_objectives, num_objectives
from multiobjective_branin_currinexp  import branin, currin_exp


def main():
  """ Main function. """
  # First Specify all parameters.
  # See examples/synthetic/multiobjective_hartmann/in_code_demo.py for speciying
  # domain via a JSON file.
  domain_vars = [{'type': 'float', 'min': -5, 'max': 10, 'dim': 1},
                 {'type': 'float',  'min': 0, 'max': 15, 'dim': 1},
                ]
  config_params = {'domain': domain_vars}
  config = load_config(config_params)

  # Specify objectives -- either of the following options could work.
  # 1. compute_objectives returns a list of objective values, num_objectives is the number
  # of objectives. This has to be a 2-tuple.
  moo_objectives = (compute_objectives, num_objectives)
  # 2. Specify each function separately. This has to be a list.
#   moo_objectives = [branin, currin_exp]

  # Optimise
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(
                                               moo_objectives, config.domain,
                                               max_num_evals, config=config)
  print(pareto_opt_pts)
  print(pareto_opt_vals)


if __name__ == '__main__':
  main()

