# dragonfly

dragonfly is a python library for scalable Bayesian optimisation.

Bayesian optimisation is used for optimising black-box functions whose evaluations are
usually expensive.
Beyond vanilla optimisation techniques, dragonfly provides an array of tools to 
scale up Bayesian optimisation to expensive large scale problems.
These include features/functionality that are especially suited for,
high dimensional optimisation (optimising for a large number of variables),
parallel evaluations in synchronous or asynchronous settings (conducting multiple
evaluations in parallel),
and
multi-fidelity optimisation (using cheap approximations to speed up the optimisation
process).

By default, dragonfly *maximises* functions.
To minimise a function, simply pass the negative of the function.
dragonfly is compatible with python2.7 and python3 and has been tested on Linux and
Mac OS platforms.

For questions and bug reports please email kandasamy@cs.cmu.edu


## Installation

* Download the package.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
```

* Source the set up file in the parent directory of dragonfly.
```bash
$ source set_up
```

* Build the direct fortran library. For this `cd` into `utils/direct_fortran` and run
  `bash make_direct.sh`. You will need a fortran compiler such as gnu95. Once this is
  done, you can run `simple_direct_test.py` to make sure that it was installed correctly.

**Requirements:**
Dragonfly requires standard python packages such as `numpy`, `scipy`, and `future` (for
python2/3 compatibility). They can be pip installed via
`$ pip install <package-name>`.

**Testing the Installation**:
To test the installation, run ```bash run_all_tests.sh```. Some of the tests are
probabilistic and could fail at times. If this happens, run the same test several times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simpy via `python unittest_xxx.py`.

## Getting started

Dragonfly can be
used directly in the command line by calling
[`dragonfly.py`](dragonfly.py)
or be imported in python code via the `maximise_function` function in
[`maximise_function.py`](maximise_function.py).
To help get started, we have provided some demos in the `demos` directory.

**Via command line**:

To use dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
We have demonstrated these using the
[`branin`](https://www.sfu.ca/~ssurjano/branin.html) function in the
[`demos`](demos) directory.
The function is defined in 
[`demos/branin.py`](demos/branin/branin.py).
The name of this file and the domain should be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
We have demonstrated these options in
[`demos/branin/config.json`](demos/branin/config.json) and
[`demos/branin/config.pb`](demos/branin/config.pb).
Then, specify the optimisation parameters in an options file, in the format shown in
[`demos/branin/options.txt`](demos/branin/options.txt).

If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The branin demo can be run via following commands:
```bash
$ python dragonfly.py --config demos/branin_json/config.json --options demos/branin_json/options.txt
$ python dragonfly.py --config demos/branin_json/config.pb --options demos/branin_json/options.txt
```

**In python code**:

You can import the main API in python code via,
```python
from dragonfly.maximise_function import maximise_function
...
max_val, max_pt = maximise_function(func, domain, max_capital)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be maximised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints.

[`demos/branin/in_code_demo.py`](demos/branin/in_code_demo.py)
demonstrates the use case for the branin function.
To execute this file, simply run
```bash
$ python demos/branin/in_code_demo.py
```

### Citation
If you use any part of this code in your work, please cite
[this manuscript](http://www.cs.cmu.edu/~kkandasa/docs/proposal.pdf).

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/LICENSE.txt).

"Copyright 2018 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu
