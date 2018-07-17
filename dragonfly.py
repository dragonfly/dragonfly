"""
  Command line too for GP Bandit Optimisation.
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu

  Usage:
  python dragonfly.py --config <config.json or config.pb> --options <options.txt>
"""

# pylint: disable=relative-import
# pylint: disable=invalid-name

from __future__ import print_function
import os
import imp

# Local
from opt.gp_bandit import get_all_gp_bandit_args_from_gp_args
from gp.euclidean_gp import euclidean_gp_args
from utils.option_handler import get_option_specs, load_options
from parser.config_parser import config_parser
from maximise_function import maximise_function

dragonfly_args = [ \
  get_option_specs('config', False, None, 'Path to the json or pb config file. '),
  get_option_specs('options', False, None, 'Path to the options file. '),
  get_option_specs('max_capital', False, None,
    'Maximum capital to be used in the experiment. '),
  get_option_specs('budget', False, None,
    'The budget of evaluations. If max_capital is none, will use this as max_capital.'),
                 ]

def main():
  """
    Maximizes a function given a config file containing the hyperparameters and the
    corresponding domain bounds.
  """
  # Loading Options
  euc_gpb_args = get_all_gp_bandit_args_from_gp_args(euclidean_gp_args)
  options = load_options(euc_gpb_args + dragonfly_args, cmd_line=True)
  if options.config is None:
    raise ValueError('Config file is required.')

  # Parsing config file
  expt_dir = os.path.dirname(os.path.abspath(os.path.realpath(options.config)))
  if not os.path.exists(expt_dir):
    raise ValueError("Experiment directory does not exist.")
  prob, parameters = config_parser(options.config)
  obj = imp.load_source(prob['name'], os.path.join(expt_dir, prob['name'] + '.py'))

  options.capital_type = 'return_value'
  if prob['method'] == 'slice' or prob['method'] == 'nuts':
    options.gpb_hp_tune_criterion = 'post_sampling'
    options.gpb_post_hp_tune_method = prob['method']
  if options.max_capital is None:
    if options.budget is None:
      raise ValueError('Specify the budget in budget or max_capital.')
    options.max_capital = options.budget
  options.max_capital = float(options.max_capital)

  # Domain bounds
  domain_bounds = [None] * len(parameters)
  for param in parameters:
    domain_bounds[param['order']] = [param['min'], param['max']]

  opt_val, opt_pt = maximise_function(obj.main, domain_bounds, options=options,
                                      num_workers=prob['num_workers'],
                                      max_capital=options.max_capital)
  print('Optimum Value in %d evals: %0.4f'%(options.max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))


if __name__ == '__main__':
  main()

