"""
  A demo demonstrating the use case for random forest regression and gradient boosted
  regression.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=relative-import


from __future__ import print_function
from dragonfly import minimise_function, minimise_multifidelity_function
from dragonfly.exd.cp_domain_utils import load_config_file
# Problem imports
from naval_gbr_mf import objective as naval_gbr_mf_obj
from naval_gbr_mf import cost as naval_gbr_cost
from naval_gbr import objective as naval_gbr_obj
from news_rfr_mf import objective as news_rfr_mf_obj
from news_rfr_mf import cost as news_rfr_cost
from news_rfr import objective as news_rfr_obj


# The problem
PROBLEM = 'naval_gbr'
PROBLEM = 'news_rfr'

# Multi-fidelity or not
IS_MF = True
IS_MF = False

# Capital
MAX_CAPITAL = 6 * 60 * 60 # 6 Hours

# This dictionary defines a map from the problem description (dataset+method, mf) to the
# problem parameters (objective, configuration_file, fidelity_cost_function).
_CHOOSER_DICT = {
  ('naval_gbr', True): (naval_gbr_mf_obj, 'config_naval_gbr_mf.json', naval_gbr_cost),
  ('naval_gbr', False): (naval_gbr_obj, 'config_naval_gbr.json', None),
  ('news_rfr', True): (news_rfr_mf_obj, 'config_news_rfr_mf.json', news_rfr_cost),
  ('news_rfr', False): (news_rfr_obj, 'config_news_rfr.json', None),
  }

LOG_FILE = '%s_mf%d'%(PROBLEM, int(IS_MF))

def main():
  """ Main Function. """
  # Choose which objective to minimise and the configuration file
  objective_to_min, config_file, fidel_cost_func = _CHOOSER_DICT[(PROBLEM, IS_MF)]
  config = load_config_file(config_file)
  log_stream = open(LOG_FILE, 'w')
  # Call the optimiser
  if IS_MF:
    opt_val, opt_pt, history = minimise_multifidelity_function(objective_to_min,
      config.fidel_space, config.domain, config.fidel_to_opt, fidel_cost_func,
      MAX_CAPITAL, capital_type='realtime', config=config, reporter=log_stream)
  else:
    opt_val, opt_pt, history = minimise_function(objective_to_min, config.domain,
      MAX_CAPITAL, capital_type='realtime', config=config, reporter=log_stream)
  # Print out result
  log_stream.close()
  print('Optimum Value found in %02.f time (%d evals): %0.4f'%(
        MAX_CAPITAL, len(history.curr_opt_points), opt_val))
  print('Optimum Point: %s.'%(str(opt_pt)))


if __name__ == '__main__':
  main()

