"""
  Reads either pickle or mat files and plots the results.
  -- syiblet@andrew.cmu.edu
  -- kvysyara@andrew.cmu.edu

  Usage:
  python plotting.py --filelist <file containing list of pickle or mat file paths>
  python plotting.py --file     <pickle or mat file path>
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals

import os
import pickle
import argparse
import warnings
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

def rgba(red, green, blue, a):
  '''rgba: generates matplotlib compatible rgba values from html-style rgba values
  '''
  return (red / 255.0, green / 255.0, blue / 255.0, a)

def hex(hexstring):
  '''hex: generates matplotlib-compatible rgba values from html-style hex colors
  '''
  if hexstring[0] == '#':
    hexstring = hexstring[1:]
  red = int(hexstring[:2], 16)
  green = int(hexstring[2:4], 16)
  blue = int(hexstring[4:], 16)
  return rgba(red, green, blue, 1.0)

def transparent(red, green, blue, _, opacity=0.5):
  '''transparent: converts a rgba color to a transparent opacity
  '''
  return (red, green, blue, opacity)


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

def get_plot_info(
    meth_curr_opt_vals,
    cum_costs,
    meth_costs,
    grid_pts,
    outlier_frac,
    init_opt_vals
):
  """generates means and standard deviation for the method's output
  """
  num_experiments = len(meth_curr_opt_vals)
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    idx = np.where(meth_curr_opt_vals == '-')
  if idx[0].size != 0:
    num_experiments = idx[0][0]

  outlier_low_idx = max(np.round(outlier_frac * num_experiments), 1)
  outlier_high_idx = min(
      num_experiments,
      int(num_experiments  - np.rint(outlier_frac * num_experiments))
  )

  inlier_idx = np.arange(outlier_low_idx, outlier_high_idx)
  num_grid_pts = len(grid_pts)
  grid_vals = np.zeros((num_experiments, num_grid_pts))

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
    grid_vals[exp_iter, :] = np.maximum.accumulate(interp)


  sorted_grid_vals = np.sort(grid_vals, axis=0)
  inlier_grid_vals = sorted_grid_vals[inlier_idx, :]

  def mean_and_std(arr1d):
    """ Returns mean and standard deviation."""
    finite_arr1d = arr1d[np.isfinite(arr1d)]
    if finite_arr1d.size / arr1d.size >= 0.4:
      return np.array([np.mean(finite_arr1d), np.std(finite_arr1d) / np.sqrt(arr1d.size)])
    return np.array([np.NaN] * 2)

  res = np.apply_along_axis(mean_and_std, 0, inlier_grid_vals)
  return (res[0, :], res[1, :])

def gen_curves(
    plot_order,
    plot_legends,
    results,
    x_label,
    y_label,
    plot_markers,
    plot_line_markers,
    plot_colors,
    x_bounds=None,
    outlier_frac=0.1,
    set_legend=True,
    log_y=False,
    log_x=False,
    plot_title=None,
    study_name=None,
    num_workers=None,
    time_distro_str=None,
    fill_error=False,
    err_bar_freq=5,
    plot_type='plot'
):
  # pylint: disable=too-many-arguments
  # pylint: disable=too-many-branches
  # pylint: disable=unused-argument
  # pylint: disable=unused-variable
  """Plots the curves given the experiment result data
  """

  NUM_GRID_PTS = 100
  NUM_ERR_BARS = 10
  LINE_WIDTH = 2

  num_methods, num_experiments = results['curr_opt_vals'].shape
  methods = [str(method).strip() for method in results['methods']]

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

  plot_means = np.zeros((num_methods, NUM_GRID_PTS))
  plot_stds = np.zeros((num_methods, NUM_GRID_PTS))

  if 'smac' in methods or 'hyperopt' in methods or 'spearmint' in methods or \
     'gpyopt' in methods:
    init_opt_vals = None
  else:
    init_opt_vals = []
    for i in range(num_experiments):
      init_opt_vals.append(np.amax(results['prev_eval_vals'][-1, :][i]))

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
        init_opt_vals
    )

    plot_means[i, :] = meth_plot_mean
    plot_stds[i, :] = meth_plot_std

  err_bar_pts = grid_pts[err_bar_idxs]
  err_bar_means = plot_means[:, err_bar_idxs]
  err_bar_stds = plot_stds[:, err_bar_idxs]

  for i, method in enumerate(methods):
    plot_idx = plot_order.index(method)
    if plot_type == 'plot':
      plt.plot(
          grid_pts,
          plot_means[i, :],
          marker=plot_markers[plot_idx],
          color=plot_colors[plot_idx],
          linewidth=LINE_WIDTH,
          label=plot_legends[plot_idx],
      )
    elif plot_type == 'loglog':
      plt.loglog(
          grid_pts,
          plot_means[i, :],
          marker=plot_markers[plot_idx],
          color=plot_colors[plot_idx],
          linewidth=LINE_WIDTH,
          label=plot_legends[plot_idx],
      )
    elif plot_type == 'semilogy':
      plt.semilogy(
          grid_pts,
          plot_means[i, :],
          marker=plot_markers[plot_idx],
          color=plot_colors[plot_idx],
          linewidth=LINE_WIDTH,
          label=plot_legends[plot_idx],
      )
    elif plot_type == 'semilogx':
      plt.semilogx(
          grid_pts,
          plot_means[i, :],
          marker=plot_markers[plot_idx],
          color=plot_colors[plot_idx],
          linewidth=LINE_WIDTH,
          label=plot_legends[plot_idx],
      )

    if not fill_error:
      plt.errorbar(
          grid_pts[plot_idx%err_bar_freq::err_bar_freq],
          plot_means[i, plot_idx%err_bar_freq::err_bar_freq],
          plot_stds[i, plot_idx%err_bar_freq::err_bar_freq],
          marker=plot_markers[plot_idx],
          color=plot_colors[plot_idx],
          linewidth=LINE_WIDTH,
          fmt='o',
      )
    else:
      plt.fill_between(
          grid_pts,
          plot_means[i, :] - plot_stds[i, :],
          plot_means[i, :] + plot_stds[i, :],
          color=transparent(*plot_colors[plot_idx], opacity=0.3),
      )

  if plot_title is not None:
    plt.suptitle(plot_title)
  plt.title('{}; M={}; {}'.format(study_name, num_workers,
                                  time_distro_str).replace('_', '-'))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc=2)
  plt.show()

def load_results(file_paths):
  """ Concatenates the results from multiple files. """
  results = read_results(file_paths[0])
  ignore = ['study_name', 'num_workers', 'time_distro_str', 'max_capital', 'methods',
            'true_maxval', 'num_jobs_per_worker', 'domain_type']
  for i in range(1, len(file_paths)):
    results_i = read_results(file_paths[i])
    for key in list(results_i.keys()):
      if key in ignore:
        continue
      elif isinstance(results_i[key], np.ndarray):
        if results_i[key].ndim == 2:
          results[key] = np.concatenate((results[key], results_i[key]), axis=1)
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

def plot_results():
  """ plots the results using matplotlib. """
  options = get_options()
  if options.filelist != '':
    file_paths = get_file_paths(options.filelist)
  elif options.file != '':
    file_paths = [os.path.realpath(os.path.abspath(options.file))]
  else:
    raise ValueError('Missing Filelist.')
  results = load_results(file_paths)
  x_label = 'Time'
  y_label = 'Max Value'
  plot_title = options.title
  #plot_order = ['ucb', 'add_ucb', 'mf_ucb', 'add_mf_ucb']
  #plot_legends = ['GP-UCB', 'Add-GP-UCB', 'MF-GP-UCB', 'Dragonfly']
  plot_order = ['ml', 'post_sampling', 'ml+post_sampling', 'rand']
  plot_legends = ['ml', 'post_sampling', 'ml+post_sampling', 'rand']
  plot_markers = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
  plot_line_markers = ['--', ':', '-.', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

  plot_colors = [
      hex('#FF4136'), #red
      hex('#0074D9'), #blue
      hex('#2ECC40'), #green
      hex('#FF851B'), #orange
      hex('#01FF70'), #lime
      hex('#F012BE'), #fuchsia
      hex('#B10DC9'), #purple
      hex('#3D9970'), #olive
      hex('#39CCCC'), #teal
      hex('#7FDBFF'), #aqua
      hex('#85144B'), #maroon
      hex('#FFDC00'), #yellow
      # hex('#001F3F'), #navy
      # hex('#FFFFFF'), #white
      # hex('#DDDDDD'), #silver
      # hex('#AAAAAA'), #gray
      # hex('#111111'), #black
  ]

  x_bounds = []
  set_legend = False
  outlier_frac = 0.0

  return gen_curves(
      plot_order,
      plot_legends,
      results,
      x_label,
      y_label,
      plot_markers,
      plot_line_markers,
      plot_colors,
      x_bounds=x_bounds,
      outlier_frac=outlier_frac,
      set_legend=set_legend,
      log_y=True,
      plot_title=plot_title,
      study_name=str(np.asscalar(results['study_name'])),
      num_workers=str(np.asscalar(results['num_workers'])),
      time_distro_str=str(np.asscalar(results['time_distro_str'])),
      plot_type=options.type
  )

def get_options():
  """ Given a list of options, this reads them from the command line and returns
      a namespace with the values.
  """
  parser = argparse.ArgumentParser(description='Plotting.')
  parser.add_argument('--file', default='', help='File path of single plot file.')
  parser.add_argument('--filelist', default='', help='File name containing file paths.')
  parser.add_argument('--type', default='plot', help='Type of plot -- default is plot, \
                      other options are semilogy, loglog, semilogx.')
  parser.add_argument('--title', help='Title of plot.')
  options = parser.parse_args()

  return options

if __name__ == '__main__':
  plot_results()
