# dragonfly
An open source python library for scalable Bayesian optimisation.
## '#"$_-*/*
(*/*')#"$_-(*/dragonfly_build-run_README.m_Config.py-to-run.Q#*)
<img src="https://dragonfly.github.io/images/dragonfly_bigwords.png"/>

"*/*"---("*/*")
Dragonfly is an open source python library for scalable Bayesian optimisation.
Bayesian optimisation is used for optimising black-box functions whose evaluations are
usually expensive. Beyond vanilla optimisation techniques, Dragonfly provides an array of tools to
scale up Bayesian optimisation to expensive large scale problems.
These include features/functionality that are especially suited for
high dimensional optimisation (optimising for a large number of variables),
parallel evaluations in synchronous or asynchronous settings (conducting multiple
evaluations in parallel), multi-fidelity optimisation (using cheap approximations
to speed up the optimisation process), and multi-objective optimisation (optimising
multiple functions simultaneously).
Dragonfly is compatible with Python2 (>= 2.7) and Python3 (>= 3.5) and has been tested
on Linux, macOS, and Windows platforms.
For documentation, installation, and a getting started guide, see our
[readthedocs page](https://dragonfly-opt.readthedocs.io). For more details, see
our [paper](https://arxiv.org/abs/1903.06694).
&nbsp;
## Installation
See 
[here](https://dragonfly-opt.readthedocs.io/en/master/install/)
for detailed instructions on installing Dragonfly and its dependencies.
**Quick Installation:**
If you have done this kind of thing before, you should be able to install
Dragonfly via `pip`.
```bash
$ sudo apt-get install python-dev python3-dev gfortran # On Ubuntu/Debian
$ pip install numpy
$ pip install dragonfly-opt -v
```
**Testing the Installation**:
You can import Dragonfly in python to test if it was installed properly.
If you have installed via source, make sure that you move to a different directory 
 to avoid naming conflicts.
```bash
$ python
>>> from dragonfly import minimise_function
>>> # The first argument below is the function, the second is the domain, and the third is the budget.
>>> min_val, min_pt, history = minimise_function(lambda x: x ** 4 - x**2 + 0.1 * x, [[-10, 10]], 10);  
...
>>> min_val, min_pt
(-0.32122746026750953, array([-0.7129672]))
```
Due to stochasticity in the algorithms, the above values for `min_val`, `min_pt` may be
different. If you run it for longer (e.g.
`min_val, min_pt, history = minimise_function(lambda x: x ** 4 - x**2 + 0.1 * x, [[-10, 10]], 100)`),
you should get more consistent values for the minimum. 
If the installation fails or if there are warning messages, see detailed instructions
[here](https://dragonfly-opt.readthedocs.io/en/master/install/).
&nbsp;
## Quick Start
Dragonfly can be
used directly in the command line by calling
[`dragonfly-script.py`](bin/dragonfly-script.py)
or be imported in python code via the `maximise_function` function in the main library
or in <em>ask-tell</em> mode.
To help get started, we have provided some examples in the
[`examples`](examples) directory.
See our readthedocs getting started pages
([command line](https://dragonfly-opt.readthedocs.io/en/master/getting_started_cli/),
[Python](https://dragonfly-opt.readthedocs.io/en/master/getting_started_py/),
[Ask-Tell](https://dragonfly-opt.readthedocs.io/en/master/getting_started_ask_tell/))
for examples and use cases.
**Command line**:
Below is an example usage in the command line.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example.txt
```
**In Python code**:
The main APIs for Dragonfly are defined in
[`dragonfly/apis`](dragonfly/apis).
For their definitions and arguments, see
[`dragonfly/apis/opt.py`](dragonfly/apis/opt.py) and
[`dragonfly/apis/moo.py`](dragonfly/apis/moo.py).
You can import the main API in python code via,
```python
from dragonfly import minimise_function, maximise_function
func = lambda x: x ** 4 - x**2 + 0.1 * x
domain = [[-10, 10]]
max_capital = 100
min_val, min_pt, history = minimise_function(func, domain, max_capital)
print(min_val, min_pt)
commit db2910e
@MoneyMan573("*/*")
 
 

© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
