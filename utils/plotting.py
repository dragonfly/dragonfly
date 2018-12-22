"""
  Reads either pickle or mat files and plots the results.
  -- syiblet@andrew.cmu.edu
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu

  Usage:
  python plotting.py --filelist <file containing list of pickle or mat file paths>
  python plotting.py --file     <pickle or mat file path>
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import os
import numpy as np
# Local
# from utils.plot_utils import COLOURS, plot_results, get_plot_options, \
#                              load_results, get_file_paths
from utils.plot_utils_duplicate import COLOURS, plot_results, get_plot_options, \
                                       load_results, get_file_paths


syn_funcs = {'hartmann3': (3.86278, 3),
             'hartmann6': (3.322368, 6),
             'currin_exp': (13.7986850, 2),
             'branin': (-0.39788735773, 2),
             'borehole': (309.523221, 8),
             'park1': (25.5872304, 4),
             'park2': (5.925698, 4)
            }

def get_true_maxval(study_name):
  ''' Returns true max values for high dim synthetic functions. '''
  name = study_name.split('-')[0]
  if name not in syn_funcs:
    return None
  if len(study_name.split('-')) == 1:
    return syn_funcs[name][0]
  group_dim = int(study_name.split('-')[1])
  max_val, domain_dim = syn_funcs[name]
  num_groups = int(group_dim/domain_dim)
  return max_val*num_groups


def main():
  """ Main function. """
#   # CP Stuff
  plot_order = ['rand', 'ga', 'hyperopt', 'smac', 'gpyopt', 'dragonfly', 'dragonfly-mf']
#   plot_order = ['rand', 'ga', 'hyperopt', 'smac', 'gpyopt', 'dragonfly']
#   plot_order = ['rand', 'ga', 'dragonfly', 'dragonfly-mf']
#   plot_order = ['rand', 'ga', 'dragonfly']

  # Euclidean Stuff
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'smac', 'spearmint', 'gpyopt',
#                 'dragonfly', 'dragonfly-mf']
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'smac', 'spearmint', 'gpyopt', 'dragonfly']

#   plot_order = ['ucb', 'add_ucb', 'mf_ucb', 'add_mf_ucb']
#   plot_order = ['ml', 'post_sampling', 'ml+post_sampling', 'rand']
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'smac', 'spearmint', 'gpyopt',
#                 'ensemble-dfl', 'adaptive-ensemble-dfl', 'dragonfly']
#   plot_order = ['ei', 'ucb', 'ttei', 'ts', 'add_ucb', 'ei-ucb-ttei-ts']
#   plot_order = ['esp-se', 'esp-matern', 'add-se', 'add-matern', 'se', 'matern', 'rand']
#   plot_order = ['mf-se', 'mf-expdecay', 'rand']
#   plot_order = ['add-ei', 'add-ucb', 'add-add_ucb', 'esp-ucb', 'esp-ei']
#   plot_order = ['ei', 'ucb', 'ttei', 'ts', 'add_ucb', 'ei-ucb-ttei-ts']


  # Load options and results
  options = get_plot_options()
  if options.filelist != '':
    file_paths = get_file_paths(options.filelist)
  elif options.file != '':
    file_paths = [os.path.realpath(os.path.abspath(options.file))]
  else:
    raise ValueError('Missing Filelist.')
  to_plot_legend = False
  results = load_results(file_paths)
  # Ancillary info for the plot
  study_name = str(np.asscalar(results['study_name']))
  true_maxval = get_true_maxval(study_name)
  x_label = 'Number of Evaluations'
  y_label = 'Simple Regret'
  y_bounds = None
  # title
  if not hasattr(options, 'title') or options.title is None:
    print study_name
    if study_name == 'borehole':
      plot_title = r'Borehole $(d=8)$'
      to_plot_legend = True
      options.legend_location = 4
    elif study_name == 'hartmann3-18':
      plot_title = r'Hartmann3$\times$6 $(d=18)$'
    elif study_name == 'park2-24':
      plot_title = r'Park2$\times$6 $(d=24)$'
    # Non-euclidean domains
    elif study_name == 'borehole_6':
      plot_title = r'Borehole_6 $(d=8)$'
      to_plot_legend = True
      options.legend_location = 4
    elif study_name == 'hartmann6_4':
      plot_title = r'Hartmann6_4 $(d=6)$'
    elif study_name == 'park2_3':
      plot_title = r'Park2_3 $(d=4)$'
    elif study_name == 'park2_4':
      plot_title = r'Park2_4 $(d=4)$'
      y_bounds = [-6, -3.6345]
    else:
      plot_title = ''
  # Method legend dictionary
  method_legend_colour_marker_dict = {
    # Packages
    "rand": {'legend':'RAND', 'colour':COLOURS['black'], 'marker':'s', 'linestyle':'-'},
    "ga": {'legend':'EA', 'colour':COLOURS['olive'], 'marker':'*', 'linestyle':'-'},
    "pdoo": {'legend':'PDOO', 'colour':COLOURS['red'], 'marker':'>', 'linestyle':'-'},
    "hyperopt": {'legend':'HyperOpt', 'colour':COLOURS['orange'], 'marker':'1',
                 'linestyle':'-'},
    "gpyopt": {'legend':'GPyOpt', 'colour':COLOURS['maroon'], 'marker':'+',
               'linestyle':'-'},
    "smac": {'legend':'SMAC', 'colour':COLOURS['magenta'], 'marker':'x',
             'linestyle':'-'},
    "spearmint": {'legend':'Spearmint', 'colour':COLOURS['green'], 'marker':'^',
                  'linestyle':'-'},
    "dragonfly": {'legend':'Dragonfly', 'colour':COLOURS['blue'], 'marker':'o',
                  'linestyle':'-'},
    "dragonfly-mf": {'legend':'Dragonfly+MF', 'colour':COLOURS['cyan'], 'marker':'d',
                     'linestyle':'-'},
    # Custom methods
    "ml": {'legend':'ML', 'colour':COLOURS['red'], 'marker':'>', 'linestyle':'-'},
    "post_sampling": {'legend':'PS', 'colour':COLOURS['green'], 'marker':'s',
                      'linestyle':'-'},
    "ml-post_sampling": {'legend':'ML+PS', 'colour':COLOURS['blue'], 'marker':'.',
                         'linestyle':'-'},
  }
  plot_results(results, plot_order, method_legend_colour_marker_dict, x_label, y_label,
               to_plot_legend=to_plot_legend, true_maxval=true_maxval,
               plot_title=plot_title, options=options)


if __name__ == '__main__':
  main()

