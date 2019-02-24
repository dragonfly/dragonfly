"""
  Utilities for plotting.
  -- syiblet@andrew.cmu.edu
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

NUM_GRID_PTS = 100
NUM_ERR_BARS = 10
LINE_WIDTH = 2
MARKER_SIZE = 6
AXIS_FONT_SIZE = 13
TITLE_FONT_SIZE = 18


def rgba(red, green, blue, a):
  '''rgba: generates matplotlib compatible rgba values from html-style rgba values
  '''
  return (red / 255.0, green / 255.0, blue / 255.0, a)

def transparent(red, green, blue, _, opacity=0.5):
  '''transparent: converts a rgba color to a transparent opacity
  '''
  return (red, green, blue, opacity)

def to_hex(hexstring):
  '''to_hex: generates matplotlib-compatible rgba values from html-style hex colours
  '''
  if hexstring[0] == '#':
    hexstring = hexstring[1:]
  red = int(hexstring[:2], 16)
  green = int(hexstring[2:4], 16)
  blue = int(hexstring[4:], 16)
  return rgba(red, green, blue, 1.0)

COLOURS = {
  'red': to_hex('#FF4136'),
  'blue': to_hex('#0074D9'),
  'cyan': to_hex('#00FFFF'),
  'magenta': to_hex('#FF00FF'),
  'green': to_hex('#2ECC40'),
  'orange': to_hex('#FF851B'),
  'lime': to_hex('#01FF70'),
  'fuchsia': to_hex('#F012BE'),
  'purple': to_hex('#B10DC9'),
  'olive': to_hex('#3D9970'),
  'teal': to_hex('#39CCCC'),
  'aqua': to_hex('#7FDBFF'),
  'maroon': to_hex('#85144B'),
  'yellow': to_hex('#FFDC00'),
  'navy': to_hex('#001F3F'),
  'white': to_hex('#FFFFFF'),
  'silver': to_hex('#DDDDDD'),
  'grey': to_hex('#AAAAAA'),
  'black': to_hex('#111111'),
}


def generate_legend_marker_colour_orders(plot_order, method_legend_colour_marker_dict):
  """ Generates a list of legends, colours, and markers based off the dictionary. """
  plot_legends = [method_legend_colour_marker_dict[meth]['legend'] for meth in plot_order]
  plot_colours = [method_legend_colour_marker_dict[meth]['colour'] for meth in plot_order]
  plot_markers = [method_legend_colour_marker_dict[meth]['marker'] for meth in plot_order]
  plot_linestyles = [method_legend_colour_marker_dict[meth]['linestyle']
                     for meth in plot_order]
  return plot_legends, plot_colours, plot_markers, plot_linestyles


def gen_curves(
    plot_order,
    results,
    x_label,
    y_label,
    plot_legends,
    plot_colours,
    plot_markers,
    plot_linestyles,
    options,
    x_bounds=None,
    y_bounds=None,
    outlier_frac=0.1,
    log_y=False,
    log_x=False,
    plot_title=None,
    study_name=None,
    num_workers=None,
    time_distro_str=None,
    fill_error=False,
    err_bar_freq=5,
    plot_type='semilogy',
    to_plot_legend=True,
    true_maxval=None,
):
  # pylint: disable=too-many-arguments
  # pylint: disable=too-many-branches
  # pylint: disable=too-many-statements
  # pylint: disable=unused-argument
  # pylint: disable=unused-variable
  """ Plots the curves given the experiment results.
  """

  num_methods, num_experiments = results['curr_opt_vals'].shape
  if true_maxval is None:
    true_maxval = np.asscalar(results['true_maxval'])

  if true_maxval is None or not np.isfinite(true_maxval):
    to_plot_regret = False
  else:
    to_plot_regret = True
    plot_type = 'semilogy'
    y_label = 'Regret'

  methods = [str(method).strip() for method in results['methods']]
  # Exclude incomplete experiments if present in any of the methods
  for i in range(num_methods):
    for vals in results['curr_opt_vals'][i, num_experiments::-1]:
      if isinstance(vals, str) or vals.dtype == np.dtype('<U1'):
        num_experiments = num_experiments - 1

  if x_bounds is None or x_bounds == []:
    x_bounds = [0.0, np.asscalar(results['max_capital'])]

  if log_x:
    grid_pts = np.logspace(
        np.log10(x_bounds[0]),
        np.log10(x_bounds[1]),
        num=NUM_GRID_PTS
    )
  else:
    grid_pts = np.linspace(
        x_bounds[0],
        x_bounds[1],
        num=NUM_GRID_PTS
    )

  err_bar_idx_half_gap = 0.5 * NUM_GRID_PTS / NUM_ERR_BARS
  err_bar_idxs = np.round(np.linspace(
      err_bar_idx_half_gap,
      NUM_GRID_PTS - err_bar_idx_half_gap,
      num=NUM_ERR_BARS
  )).astype(np.int)

  unordered_plot_means = np.zeros((num_methods, NUM_GRID_PTS))
  unordered_plot_stds = np.zeros((num_methods, NUM_GRID_PTS))
  init_opt_vals = None

  for i in range(num_methods):
    meth_curr_opt_vals = results['curr_true_opt_vals'][i, :]
    meth_costs = results['query_eval_times'][i, :]
    cum_costs = results['query_receive_times'][i, :]
    meth_plot_mean, meth_plot_std = get_plot_info(
        meth_curr_opt_vals,
        cum_costs,
        meth_costs,
        grid_pts,
        outlier_frac,
        init_opt_vals,
        num_experiments
    )

    if to_plot_regret:
      unordered_plot_means[i, :] = true_maxval - meth_plot_mean
    else:
      unordered_plot_means[i, :] = meth_plot_mean
    unordered_plot_stds[i, :] = meth_plot_std

  # re-order plot_means
  plot_means = np.zeros((num_methods, NUM_GRID_PTS))
  plot_stds = np.zeros((num_methods, NUM_GRID_PTS))
  for plot_idx, method in enumerate(plot_order):
    saved_order = methods.index(method)
    plot_means[plot_idx, :] = unordered_plot_means[saved_order, :]
    plot_stds[plot_idx, :] = unordered_plot_stds[saved_order, :]

  # Print out some statistics about plot_means
  all_plot_vals = plot_means.flatten()
  all_plot_vals = all_plot_vals[np.isfinite(all_plot_vals)]
  percentiles = [0.001, 0.1, 0.5, 0.9, 0.999]
  percentile_vals = np.percentile(all_plot_vals, percentiles)
  print_percentile_list = ['%0.3f:%0.4f'%(p, v) for (p, v) in
                           zip(percentiles, percentile_vals)]
  print('Percentiles:: %s.'%(',  '.join(print_percentile_list)))

  err_bar_pts = grid_pts[err_bar_idxs]
  err_bar_means = plot_means[:, err_bar_idxs]
  err_bar_stds = plot_stds[:, err_bar_idxs]

  if plot_type == 'plot':
    plot_func = plt.plot
  elif plot_type == 'loglog':
    plot_func = plt.loglog
  elif plot_type == 'semilogy':
    plot_func = plt.semilogy
  elif plot_type == 'semilogx':
    plot_func = plt.semilogx
  else:
    raise ValueError('Unknown plot function.')

  first_lines_for_legend = []

  # First the bars for the legend
#   for i, method in enumerate(methods):
  for plot_idx, method in enumerate(plot_order):
    # First plot the error bars
    curr_leg_line, = plot_func(
                       err_bar_pts,
                       err_bar_means[plot_idx],
                       marker=plot_markers[plot_idx],
                       color=plot_colours[plot_idx],
                       label=plot_legends[plot_idx],
                       linewidth=LINE_WIDTH,
                       linestyle=plot_linestyles[plot_idx],
                       markersize=MARKER_SIZE,
                       )
    first_lines_for_legend.append(curr_leg_line)
  if to_plot_legend:
    plt.legend(loc=options.legend_location, fontsize='large')
  # Now plot the whole curve
  for plot_idx, method in enumerate(plot_order):
    plot_func(
          grid_pts,
          plot_means[plot_idx, :],
          marker=',',
          color=plot_colours[plot_idx],
          linestyle=plot_linestyles[plot_idx],
          linewidth=LINE_WIDTH,
          )
  # Now do the error bars
  for plot_idx, method in enumerate(plot_order):
    if not fill_error:
      plt.errorbar(
          err_bar_pts,
          err_bar_means[plot_idx],
          err_bar_stds[plot_idx],
          marker=plot_markers[plot_idx],
          color=plot_colours[plot_idx],
          linewidth=LINE_WIDTH,
          markersize=MARKER_SIZE,
          linestyle='',
      )
    else:
      plt.fill_between(
          grid_pts,
          plot_means[i, :] - plot_stds[i, :],
          plot_means[i, :] + plot_stds[i, :],
          color=transparent(*plot_colours[plot_idx], opacity=0.3),
      )

  if plot_title is not None:
    plt.title(plot_title, fontsize=TITLE_FONT_SIZE)
#   if to_plot_regret:
#     plt.title('{}; workers={}; {}; Exps={}; Max Val={}'.format(study_name, num_workers,
#                       time_distro_str, num_experiments, true_maxval).replace('_', '-'))
#   else:
#     plt.title('{}; Workers={}; {}; Exps={}'.format(study_name, num_workers,
#                       time_distro_str, num_experiments).replace('_', '-'))
  plt.xlabel(x_label, fontsize=AXIS_FONT_SIZE)
  plt.ylabel(y_label, fontsize=AXIS_FONT_SIZE)
  if y_bounds is not None:
    plt.ylim(y_bounds)
  # Remove duplicate lines
  for leg_line in first_lines_for_legend:
    leg_line.remove()
  plt.draw()
  plt.show()


# Utilities to read and process data -----------------------------------------------
def get_plot_info(
    meth_curr_opt_vals,
    cum_costs,
    meth_costs,
    grid_pts,
    outlier_frac,
    init_opt_vals,
    num_experiments
):
  """ Generates means and standard deviation for the method's output
  """
  outlier_low_idx = int(max(np.round(outlier_frac * num_experiments), 0))
  outlier_high_idx = min(
      num_experiments,
      int(num_experiments  - np.rint(outlier_frac * num_experiments))
  )
  # Manage potential outliers
  inlier_idx = 0
  if num_experiments > 1:
    inlier_idx = np.arange(outlier_low_idx, outlier_high_idx)
  num_grid_pts = len(grid_pts)
  grid_vals = np.zeros((num_experiments, num_grid_pts))
  # Iterate through each experiment
  for exp_iter in range(num_experiments):
    if cum_costs is None:
      curr_cum_costs = np.cumsum(meth_costs[exp_iter])
    else:
      curr_cum_costs = cum_costs[exp_iter]
    if init_opt_vals is not None:
      opt_vals = np.concatenate((np.array([init_opt_vals[exp_iter]]),
                                 np.squeeze(meth_curr_opt_vals[exp_iter])), axis=0)
      curr_cum_costs = np.concatenate((np.array([0]), np.squeeze(curr_cum_costs)),
                                      axis=0)
    else:
      opt_vals = meth_curr_opt_vals[exp_iter]
    interp = np.interp(grid_pts, curr_cum_costs.flatten(), opt_vals.flatten())
    grid_vals[exp_iter, :] = np.fmax.accumulate(interp)
  sorted_grid_vals = np.sort(grid_vals, axis=0)
  inlier_grid_vals = sorted_grid_vals[inlier_idx, :]
  if num_experiments == 1:
    inlier_grid_vals = np.expand_dims(inlier_grid_vals, axis=0)
  # An internal function to get mean and std
  def mean_and_std(arr1d):
    """ Returns mean and standard deviation."""
    finite_arr1d = arr1d[np.isfinite(arr1d)]
    if finite_arr1d.size / arr1d.size >= 0.4:
      return np.array([np.mean(finite_arr1d), np.std(finite_arr1d) / np.sqrt(arr1d.size)])
    return np.array([np.NaN] * 2)
  # Return
  res = np.apply_along_axis(mean_and_std, 0, inlier_grid_vals)
  return (res[0, :], res[1, :])



# Utilities to read results from a file --------------------------------------------
def read_results(file_path):
  """reads experiment result data from a '.m' file
  :file_path: the path to the file
  :returns: a dataframe object with all the various pieces of data

  """
  if file_path.endswith('.mat'):
    results = loadmat(file_path)
  elif file_path.endswith('.p'):
    with open(file_path, 'rb') as pickleF:
      res = pickle.load(pickleF)
      pickleF.close()

    results = {}
    for key in list(res.keys()):
      if not hasattr(res[key], '__len__'):
        results[key] = np.array(res[key])
      elif isinstance(res[key], str):
        results[key] = np.array(res[key])
      elif isinstance(res[key], list):
        results[key] = np.array(res[key])
      elif isinstance(res[key], np.ndarray):
        val = np.zeros(res[key].shape, dtype=res[key].dtype)
        for idx, x in np.ndenumerate(res[key]):
          if isinstance(x, list):
            val[idx] = np.array(x)
          else:
            val[idx] = x
        results[key] = val
      else:
        results[key] = res[key]
  else:
    raise ValueError('Wrong file format. It has to be either mat or pickle file')
  return results

# Utilities for loading results and plotting ------------------------------------------
def load_results(file_paths):
  """ Concatenates the results from multiple files. """
  results = read_results(file_paths[0])
  ignore = ['study_name', 'num_workers', 'time_distro_str', 'max_capital',
            'true_maxval', 'num_jobs_per_worker', 'domain_type', 'true_argmax']
  for i in range(1, len(file_paths)):
    results_i = read_results(file_paths[i])
    diff_methods = False
    if len(results_i['methods']) != len(results['methods']):
      diff_methods = True
    elif not (results_i['methods'] == results['methods']).all():
      diff_methods = True
    for key in list(results_i.keys()):
      if key in ignore:
        continue
      elif isinstance(results_i[key], np.ndarray):
        if results_i[key].ndim == 2:
          if not diff_methods:
            results[key] = np.concatenate((results[key], results_i[key]), axis=1)
          else:
            if results[key].ndim != 2:
              results[key] = np.expand_dims(results[key], axis=0)
            _, n1 = results_i[key].shape
            _, n = results[key].shape
            if n1 < n:
              n = n1
            results[key] = np.concatenate((results[key][:, :n], results_i[key][:, :n]),
                                          axis=0)
        elif results_i[key].ndim == 0:
          continue
        else:
          results[key] = np.concatenate((results[key], results_i[key]))
  return results

def get_file_paths(fname):
  """ Read file paths from the file. """
  with open(fname) as f:
    file_paths = f.readlines()
  file_paths = [path.rstrip() for path in file_paths if path.rstrip()]
  return file_paths


def get_plot_options():
  """ Given a list of options, this reads them from the command line and returns
      a namespace with the values.
  """
  parser = argparse.ArgumentParser(description='Plotting.')
  parser.add_argument('--file', default='', help='File path of single plot file.')
  parser.add_argument('--filelist', default='', help='File name containing file paths.')
  parser.add_argument('--type', default='semilogy', help='Type of plot. Default is ' +
                      'semilogy, other options are plot, loglog, semilogx.')
  parser.add_argument('--title', help='Title of plot.')
  options = parser.parse_args()

  return options


def plot_results(results, plot_order, method_legend_colour_marker_dict, x_label,
                 y_label, x_bounds=None, y_bounds=None, to_plot_legend=True,
                 true_maxval=None, outlier_frac=0.0, plot_title=None, options=None):
  # pylint: disable=too-many-arguments
  """ Plots the results using Matplotlib. """
  if options is None:
    options = get_plot_options()
  if plot_title == None:
    plot_title = options.title
  if x_bounds is None:
    x_bounds = []
  # Get order of legends, colours, etc.
  plot_legends, plot_colours, plot_markers, plot_linestyles = \
    generate_legend_marker_colour_orders(plot_order, method_legend_colour_marker_dict)
  return gen_curves(
      plot_order,
      results,
      x_label,
      y_label,
      plot_legends,
      plot_colours,
      plot_markers,
      plot_linestyles,
      options,
      x_bounds=x_bounds,
      y_bounds=y_bounds,
      outlier_frac=outlier_frac,
      log_y=True,
      plot_title=plot_title,
      study_name=str(np.asscalar(results['study_name'])),
      num_workers=str(np.asscalar(results['num_workers'])),
      time_distro_str=str(np.asscalar(results['time_distro_str'])),
      plot_type=options.type,
      to_plot_legend=to_plot_legend,
      true_maxval=true_maxval,
  )

