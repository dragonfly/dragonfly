"""
  Main APIs and command line tool for GP Bandit Optimisation.
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu

  Usage:
  python dragonfly.py --config <config.json or config.pb> --options <options.txt>
"""

# pylint: disable=relative-import
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=redefined-variable-type

from __future__ import print_function
import os
import imp

# Local
from opt.gp_bandit import get_all_gp_bandit_args_from_gp_args
from gp.euclidean_gp import euclidean_gp_args
from utils.option_handler import get_option_specs, load_options
from parse.config_parser import config_parser
from utils.reporters import get_reporter
from exd.worker_manager import SyntheticWorkerManager
from exd.experiment_caller import EuclideanFunctionCaller, FunctionCaller
from opt.gp_bandit import EuclideanGPBandit
from exd import domains

dragonfly_args = [ \
  get_option_specs('config', False, None, 'Path to the json or pb config file. '),
  get_option_specs('options', False, None, 'Path to the options file. '),
  get_option_specs('max_capital', False, 0.0,
                   'Maximum capital to be used in the experiment. '),
  get_option_specs('budget', False, 0.0, \
      'The budget of evaluations. If max_capital is none, will use this as max_capital.'),
                 ]


def maximise_function(func, max_capital, domain=None, domain_bounds=None,
                      config=None, num_workers=1, options=None,
                      hp_tune_criterion='post_sampling',
                      hp_tune_method='slice', init_capital=None,
                      init_capital_frac=None, num_init_evals=20):
  """
    Maximizes a function given a function and domain bounds of the hyperparameters
    and returns optimal value and optimal point.
  """
  reporter = get_reporter('default')
  if options is None:
    euc_gpb_args = get_all_gp_bandit_args_from_gp_args(euclidean_gp_args)
    options = load_options(euc_gpb_args)
    options.gpb_hp_tune_criterion = hp_tune_criterion
    options.gpb_post_hp_tune_method = hp_tune_method
    options.init_capital = init_capital
    options.init_capital_frac = init_capital_frac
    options.num_init_evals = num_init_evals

  # Check for Domains
  if domain is None:
    if config is not None:
      param_spec = config_parser(options.config)
      domain = create_domain(param_spec['domain'])
    elif domain_bounds is not None:
      domain = domains.EuclideanDomain(domain_bounds)
    else:
      raise ValueError('Domain or path to config file or domain_bounds have to be given.')

  # Create worker manager and function caller
  worker_manager = SyntheticWorkerManager(num_workers, time_distro='caller_eval_cost')
  if isinstance(domain, domains.EuclideanDomain):
    func_caller = EuclideanFunctionCaller(func, domain, vectorised=False)
  else:
    func_caller = FunctionCaller(func, domain)
  # Create GPBandit opbject and run optimiser
  gpb = EuclideanGPBandit(func_caller, worker_manager, reporter=reporter, options=options)
  opt_val, opt_pt, _ = gpb.optimise(max_capital)
  opt_pt = func_caller.get_raw_domain_coords(opt_pt)
  return opt_val, opt_pt


def create_domain(parameters):
  """ Create domain object based on the bounds and kernel type. """
  list_of_domains = []
  euclidean_bounds = []
  integral_bounds = []
  discrete_bounds = []
  for param in parameters:
    bounds = [param['min'], param['max']]
    if param['type'] == 'float' and param['kernel'] == '':
      euclidean_bounds.append(bounds)
    elif param['type'] == 'float' and param['kernel'] != '':
      list_of_domains.append(domains.EuclideanDomain([bounds]))
    elif param['type'] == 'int' and param['kernel'] == '':
      integral_bounds.append(bounds)
    elif param['type'] == 'int' and param['kernel'] != '':
      list_of_domains.append(domains.IntegralDomain([bounds]))
    elif param['type'] == 'discrete' and param['kernel'] == '':
      discrete_bounds.append(bounds)
    elif param['type'] == 'discrete' and param['kernel'] != '':
      list_of_domains.append(domains.DiscreteDomain([bounds]))

  if len(euclidean_bounds) != 0:
    list_of_domains.append(domains.EuclideanDomain(euclidean_bounds))
  if len(integral_bounds) != 0:
    list_of_domains.append(domains.IntegralDomain(integral_bounds))
  if len(discrete_bounds) != 0:
    list_of_domains.append(domains.DiscreteDomain(discrete_bounds))

  if len(list_of_domains) == 1:
    return list_of_domains[0]

  domain = domains.CartesianProductDomain(list_of_domains)
  return domain


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
  param_spec = config_parser(options.config)
  exp_info = param_spec['exp_info']
  obj = imp.load_source(exp_info['name'], os.path.join(expt_dir, exp_info['name']+'.py'))

  options.capital_type = 'return_value'
  if options.max_capital == 0.0:
    if options.budget == 0.0:
      raise ValueError('Specify the budget in budget or max_capital.')
    options.max_capital = options.budget

  domain = create_domain(param_spec['domain'])
  opt_val, opt_pt = maximise_function(obj.main, domain=domain, options=options,
                                      num_workers=exp_info['num_workers'],
                                      max_capital=options.max_capital)
  print('Optimum Value in %d evals: %0.4f'%(options.max_capital, opt_val))
  print('Optimum Point: %s'%(opt_pt))


if __name__ == '__main__':
  main()

