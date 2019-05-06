"""
  A harness to load options.
  -- kandasamy@cs.cmu.edu
"""

import argparse
from copy import deepcopy

class LoadFromFile(argparse.Action):
  """ Argparse Action class to read from file. """
  def __call__(self, parser, namespace, values, option_string=None):
    with values as _file:
      options = []
      for line in _file.read().splitlines():
        if len(line) <= 2 or line[0:2] != '--':
          continue
        args_ = line.split()
        if len(args_) <= 1:
          continue
        options.extend(args_[0:2])
      parser.parse_args(options, namespace)


def get_option_specs(name, required=False, default=None, help_str='', **kwargs):
  """ A wrapper function to get a specification as a dictionary. """
  if isinstance(default, int):
    ret = {'name':name, 'required':required, 'default':default, 'help':help_str,
           'type':int}
  elif isinstance(default, float):
    ret = {'name':name, 'required':required, 'default':default, 'help':help_str,
           'type':float}
  else:
    ret = {'name':name, 'required':required, 'default':default, 'help':help_str}
  for key, value in list(kwargs.items()):
    ret[key] = value
  return ret


def _print_options(ondp, desc, reporter):
  """ Prints the options out. """
  if reporter is None:
    return
  title_str = 'Hyper-parameters for %s '%(desc)
  title_str = title_str + '-'*(80 - len(title_str))
  reporter.writeln(title_str)
  for key, value in sorted(ondp.items()):
    is_changed_str = '*' if value[0] != value[1] else ' '
    reporter.writeln('  %s %s %s'%(key.ljust(30), is_changed_str, str(value[1])))


def load_options(list_of_options, descr='Algorithm', reporter=None, cmd_line=False,
                 partial_options=None):
  """ Given a list of options, this reads them from the command line and returns
      a namespace with the values.
  """
  parser = argparse.ArgumentParser(description=descr)
  opt_names_default_parsed = {}
  for elem in list_of_options:
    opt_dict = deepcopy(elem)
    opt_name = opt_dict.pop('name')
    opt_names_default_parsed[opt_name] = [opt_dict['default'], None]
    if not opt_name.startswith('--'):
      opt_name = '--' + opt_name
    if opt_name == '--options':
      opt_dict['type'] = open
      opt_dict['action'] = LoadFromFile
    parser.add_argument(opt_name, **opt_dict)
  if cmd_line:
    args, _ = parser.parse_known_args()
  else:
    args = parser.parse_args(args=[])
  for key in opt_names_default_parsed:
    opt_names_default_parsed[key][1] = getattr(args, key)
  _print_options(opt_names_default_parsed, descr, reporter)
  # Now override with what is available in partial options
  if partial_options is not None:
    if isinstance(partial_options, dict):
      partial_options_dict = partial_options
    else:
      partial_options_dict = vars(partial_options)
    # Now override
    for key, val in partial_options_dict.items():
      setattr(args, key, val)
  return args

