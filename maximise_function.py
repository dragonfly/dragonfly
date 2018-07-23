"""
  A convenient API for GP Bandit Optimisation in Euclidean Spaces.
  For more flexibility, see opt/gp_bandit.py
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=relative-import
# pylint: disable=too-many-arguments

# Local
from opt.gp_bandit import get_all_gp_bandit_args_from_gp_args
from gp.euclidean_gp import euclidean_gp_args
from utils.option_handler import load_options
from utils.reporters import get_reporter
from exd.worker_manager import SyntheticWorkerManager
from exd.experiment_caller import EuclideanFunctionCaller
from opt.gp_bandit import EuclideanGPBandit

def maximise_function(func, domain_bounds, max_capital, options=None,
                      num_workers=1,
                      hp_tune_criterion='post_sampling',
                      hp_tune_method='slice',
                      init_capital=None,
                      init_capital_frac=None,
                      num_init_evals=20):
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
  # Create worker manager and function caller
  worker_manager = SyntheticWorkerManager(num_workers, time_distro='caller_eval_cost')
  func_caller = EuclideanFunctionCaller(func, domain_bounds, vectorised=False)
  # Create GPBandit opbject and run optimiser
  gpb = EuclideanGPBandit(func_caller, worker_manager, reporter=reporter, options=options)
  opt_val, opt_pt, _ = gpb.optimise(max_capital)
  opt_pt = func_caller.get_raw_domain_coords(opt_pt)
  return opt_val, opt_pt

