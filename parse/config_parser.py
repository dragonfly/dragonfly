"""
  Parser for json and protocol buffer files
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

import sys
import json
from collections import OrderedDict
import numpy as np

def unicode_to_str(data):
  """ Unicode to string conversion. """
  if not isinstance(data, str):
    return data.encode('utf-8')
  return data

def load_parameters(config):
  """ Parses all the parameters from json config file. """
  exp_info = {}
  exp_info['name'] = unicode_to_str(config.get('name'))
  if exp_info['name'] is None:
    raise ValueError('Experiment name is required')
  exp_info['num_trials'] = config.get('num_trials', 1)
  exp_info['num_workers'] = config.get('num_workers', 1)
  exp_info['time_distro'] = unicode_to_str(config.get('time_distro', 'const').lower())
  exp_info['results_dir'] = unicode_to_str(config.get('results_dir', 'results'))
  exp_info['method'] = unicode_to_str(config.get('method', 'slice').lower())
  exp_info['noisy_evals'] = config.get('noisy_evals', True)
  exp_info['noise_scale'] = config.get('noise_scale', 0.1)
  exp_info['reporter'] = config.get('reporter', 'default')
  exp_info['initial_pool_size'] = config.get('initial_pool_size', 20)

  parameters = []
  _parameters = config['domain']
  if isinstance(_parameters, dict):
    for key in list(_parameters.keys()):
      param = load_parameter(_parameters[key], key)
      parameters.append(param)
  elif isinstance(_parameters, list):
    for _parameter in _parameters:
      param = load_parameter(_parameter)
      parameters.append(param)
  else:
    raise ValueError('Wrong parameter type.')

  fidel_parameters = []
  _parameters = config.get('fidel_space', {})
  if isinstance(_parameters, dict):
    for key in list(_parameters.keys()):
      param = load_parameter(_parameters[key], key)
      fidel_parameters.append(param)
  elif isinstance(_parameters, list):
    for _parameter in _parameters:
      param = load_parameter(_parameter)
      fidel_parameters.append(param)
  else:
    raise ValueError('Wrong parameter type.')

  return {"exp_info" : exp_info, "domain" : parameters, "fidel_space" : fidel_parameters}

def load_parameter(parameter, key=None):
  """ Parses each parameter and return a dict """
  _name = parameter.get('name', key)
  if _name is None:
    raise ValueError('Parameter name is required')
  _type = parameter.get('type', 'float')
  _dim = parameter.get('dim', "")
  _min = parameter.get('min', -np.inf)
  _max = parameter.get('max', np.inf)
  _kernel = parameter.get('kernel', '')
  _items = parameter.get('items', '')
  if _dim != "":
    _dim = int(_dim)
  if not isinstance(_dim, (int, float, long)):
    _dim = unicode_to_str(_dim)
  param = {}
  param['name'] = unicode_to_str(_name)
  param['type'] = unicode_to_str(_type).lower()
  if param['type'] in ['float', 'int']:
    param['min'] = _min
    param['max'] = _max
  elif param['type'] == 'discrete':
    if _items == '':
      raise ValueError('List of items required')
    param['items'] = unicode_to_str(_items).split('-')
  elif param['type'] == 'discrete_numeric':
    if _items == '':
      raise ValueError('List or range of items required')
    elif ':' not in _items:
      param['items'] = [float(x) for x in unicode_to_str(_items).split('-')]
    else:
      _range = [float(x) for x in unicode_to_str(_items).split(':')]
      param['items'] = list(np.arange(_range[0], _range[2], _range[1]))
  param['kernel'] = unicode_to_str(_kernel)
  param['dim'] = _dim

  return param

def read_json(config_file):
  """ Read from json file. """
  try:
    with open(config_file, 'r') as _file:
      config = json.load(_file, object_pairs_hook=OrderedDict)
      _file.close()
  except:
    raise Exception('Error in loading config file: ' + config_file)

  return load_parameters(config)

def read_pb(config_file):
  """ Read from protocol buffer file. """
  try:
    from google.protobuf import text_format
    from google.protobuf.json_format import MessageToDict
  except ImportError:
    raise ImportError('Protocol Buffer library is not installed')

  from parse import config_pb2
  config_pb = config_pb2.Experiment()

  _file = open(config_file, "rb")
  text_format.Merge(_file.read(), config_pb)
  _file.close()

  config = MessageToDict(config_pb)
  config['fidel_space'] = config.pop('fidelSpace', {})
  return load_parameters(config)

def config_parser(config_file):
  """Reads config files and creates domain objects. """
  if config_file.endswith('.json'):
    params = read_json(config_file)
  elif config_file.endswith('.pb'):
    params = read_pb(config_file)
  else:
    raise ValueError('Wrong Config file: %s' % (config_file))

  return params

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise ValueError('Need Config File.')

  config_parser(sys.argv[1])
