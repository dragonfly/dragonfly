"""
  Parser for json and protocol buffer files
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

#pylint: disable=relative-import

import sys
import json
from collections import OrderedDict
import numpy as np

def unicode_to_str(data):
  """ Unicode to string conversion. """
  if not isinstance(data, str):
    return data.encode('utf-8')
  return data

def load_parameters_json(config):
  """ Parses all the parameters from json config file. """
  prob_params = {}
  prob_params['name'] = unicode_to_str(config.get('name'))
  if prob_params['name'] is None:
    raise ValueError('Experiment name is required')
  prob_params['num_trials'] = config.get('num_trials', 1)
  prob_params['num_workers'] = config.get('num_workers', 1)
  prob_params['time_distro'] = unicode_to_str(config.get('time_distro', 'const').lower())
  prob_params['results_dir'] = unicode_to_str(config.get('results_dir', 'results'))
  prob_params['method'] = unicode_to_str(config.get('method', 'slice').lower())
  prob_params['noisy_evals'] = config.get('noisy_evals', True)
  prob_params['noise_scale'] = config.get('noise_scale', 0.1)
  prob_params['reporter'] = config.get('reporter', 'default')
  prob_params['initial_pool_size'] = config.get('initial_pool_size', 20)

  parameters = []
  _parameters = config['parameters']
  order = 0
  for key in list(_parameters.keys()):
    _name = _parameters[key].get('name', key)
    if _name is None:
      raise ValueError('Parameter name is required')
    _type = _parameters[key].get('type', 'float')
    _dim = _parameters[key].get('dim', 1)
    _min = _parameters[key].get('min', -np.inf)
    _max = _parameters[key].get('max', np.inf)

    for i in range(_dim):
      param = {}
      if _dim == 1:
        param['name'] = unicode_to_str(_name)
      else:
        param['name'] = unicode_to_str(_name) + str(i)
      param['type'] = unicode_to_str(_type).lower()
      param['min'] = _min
      param['max'] = _max
      param['order'] = order

      parameters.append(param)
      order = order + 1

  return prob_params, parameters

def load_parameters_pb(config):
  """ Parses all the parameters from protocol buffer config file. """
  prob_params = {}
  prob_params['name'] = unicode_to_str(config.name)
  prob_params['num_trials'] = config.num_trials
  prob_params['num_workers'] = config.num_workers
  prob_params['time_distro'] = unicode_to_str(config.time_distro).lower()
  prob_params['results_dir'] = unicode_to_str(config.results_dir)
  prob_params['method'] = unicode_to_str(config.method).lower()
  prob_params['noisy_evals'] = config.noisy_evals
  prob_params['noise_scale'] = config.noise_scale
  prob_params['reporter'] = config.reporter
  prob_params['initial_pool_size'] = config.initial_pool_size

  parameters = []
  order = 0
  for var in config.variable:
    for i in range(var.dim):
      param = {}
      if var.dim == 1:
        param['name'] = unicode_to_str(var.name)
      else:
        param['name'] = unicode_to_str(var.name) + str(i)

      param['type'] = var.type

      if var.min == '-inf':
        param['min'] = -np.inf
      else:
        param['min'] = var.min

      if var.max == 'inf':
        param['max'] = np.inf
      else:
        param['max'] = var.max
      param['order'] = order

      parameters.append(param)
      order = order + 1

  return prob_params, parameters

def read_json(config_file):
  """ Read from json file. """
  try:
    with open(config_file, 'r') as _file:
      config = json.load(_file, object_pairs_hook=OrderedDict)
      _file.close()
  except:
    raise Exception('Error in loading config file: ' + config_file)

  return load_parameters_json(config)

def read_pb(config_file):
  """ Read from protocol buffer file. """
  try:
    from google.protobuf import text_format
  except ImportError:
    raise ImportError('Protocol Buffer library is not installed')

  from parser import config_pb2
  config = config_pb2.Experiment()

  _file = open(config_file, "rb")
  text_format.Merge(_file.read(), config)
  _file.close()

  return load_parameters_pb(config)

def config_parser(config_file):
  """Reads config files and creates domain objects. """
  if config_file.endswith('.json'):
    prob_params, parameters = read_json(config_file)
  elif config_file.endswith('.pb'):
    prob_params, parameters = read_pb(config_file)
  else:
    raise ValueError('Wrong Config file: %s' % (config_file))

  return prob_params, parameters

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise ValueError('Need Config File.')

  config_parser(sys.argv[1])
