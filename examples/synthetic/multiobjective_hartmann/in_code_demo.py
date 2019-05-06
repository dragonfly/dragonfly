"""
  In code demo for multiobjective_hartmann
  -- kandasamy@cs.cmu.edu
"""

from dragonfly import load_config_file, multiobjective_maximise_functions
# From current directory
# from multiobjective_hartmann  import compute_objectives, num_objectives
from multiobjective_hartmann import hartmann3_by_2_1, hartmann6, hartmann3_by_2_2


def main():
  """ Main function. """
  # First Specify the domain via a JSON configuration file.
  # See examples/synthetic/multiobjective_branin_currinexp/in_code_demo.py for speciying
  # domain directly in code without a file.
  config = load_config_file('config.json')

  # Specify objectives -- either of the following options could work. Uncomment
  # appropriately from imports and multiobjective_hartmann.py
  # 1. compute_objectives returns a list of objective values, num_objectives is the number
  # of objectives. This has to be a 2-tuple.
  # moo_objectives = (compute_objectives, num_objectives)
  # 2. Specify each function separately. This has to be a list.
  moo_objectives = [hartmann3_by_2_1, hartmann6, hartmann3_by_2_2]

  # Optimise
  max_num_evals = 100 # Optimisation budget (max number of evaluations)
  pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(
                                               moo_objectives, config.domain,
                                               max_num_evals, config=config)
  print(pareto_opt_pts)
  print(pareto_opt_vals)


if __name__ == '__main__':
  main()

