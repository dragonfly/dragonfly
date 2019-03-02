
<img src="https://dragonfly.github.io/images/dragonfly_bigwords.png"/>

---

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


&nbsp;

## Installation

**Set up:**
We recommend installation via `pip`.
In most Linux environments, it can be installed via the commands below,
depending on the version.
```bash
$ sudo apt-get install python-pip    # for Python2
$ sudo apt-get install python3-pip   # for Python3
$ pip install --upgrade pip
```
Alternatively, if you prefer to work in a Python virtual environment, `pip` is
automatically available.
If so, you need to install the appropriate version in your system.
In most Linux environments, this can be done via
`sudo apt-get install virtualenv`
if you are using Python2, or
`sudo apt-get install python3-venv`
if you are using Python3. 
You can also follow the instructions
[here](https://pip.pypa.io/en/stable/installing/),
[here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/#installing-virtualenv),
[here](https://sourabhbajaj.com/mac-setup/Python/pip.html),
[here](https://sourabhbajaj.com/mac-setup/Python/virtualenv.html),
or
[here](https://pymote.readthedocs.io/en/latest/install/windows_virtualenv.html)
for Linux, macOS and Windows environments.


The next step is recommended but not required to get started with Dragonfly.
Dragonfly uses some Fortran dependencies which require a NumPy compatible Fortran
compiler (e.g. gnu95, pg, pathf95) and the `python-dev` package.
In most Linux environments, they can be installed via
`sudo apt-get install python-dev gfortran`
if you are using Python2, or
`sudo apt-get install python3-dev gfortran`
if you are using Python3. 
These packages may already be pre-installed in your system.
If you are unable to install these packages, then you can still use Dragonfly, but
it might be slightly slower.  

You can now install Dragonfly via one of the four steps below.

&nbsp;

**1. Installation via pip (recommended):**
Installing dragonfly properly requires that numpy is already installed in the current
environment. Once that has been done, the library can be installed with pip.

```bash
$ pip install numpy
$ pip install dragonfly-opt -v
```

**2. Installation via source:**
To install via source, clone the repository and proceed as follows.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
$ cd dragonfly
$ pip install -r requirements.txt
$ python setup.py install
```


**3. Installing in a Python Virtual Environment:**
Dragonfly can be pip installed in a python virtualenv, by following the steps below.
You can similarly install via source by creating/sourcing the virtualenv and following the
steps above.
```bash
$ virtualenv env        # For Python2
$ python3 -m venv env   # For Python3
$ source env/bin/activate
(env)$ pip install numpy
(env)$ pip install git+https://github.com/dragonfly/dragonfly.git
```


**4. Using Dragonfly without Installation:**
If you prefer to not install Dragonfly in your environment, you can use it by following
the steps below.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
$ cd dragonfly
$ pip install -r requirements.txt
$ cd dragonfly/utils/direct_fortran
$ bash make_direct.sh
```
This should create a file `direct.so` in the 
[`direct_fortran`](dragonfly/utils/direct_fortran) directory.
If not,
you can still use Dragonfly, but it might be slightly slower.
Alternatively, please refer the steps above to install a Fortran compiler.
Once you have done this, you need to execute the following commands from the
[`dragonfly/dragonfly`](dragonfly) directory *every time* before using
Dragonfly.
```bash
$ HOME_PATH=$(pwd)
$ PATH=$PATH:$HOME_PATH
$ PYTHONPATH=$HOME_PATH
```



**Requirements:**
Dragonfly requires standard python packages such as `numpy`, `scipy`, `future`, and
`six` (for Python2/3 compatibility). They should be automatically installed if you follow
the above installation procedure(s). You can also manually install them by
`$ pip install <package-name>`.


&nbsp;


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
different. If you run it for longer (e.g. `min_val, min_pt, history =
minimise_function(lambda x:
x ** 4 - x**2 + 0.1 * x, [[-10, 10]], 100)`), you should get more consistent values for
the minimum. 

To run all unit tests, run ```bash run_all_tests.sh```. Some of the tests are
stochastic and could fail occasionally. If this happens, run the same test a few times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simply via `python -m unittest path.to.test.unittest_xxx`.

&nbsp;

## Getting started

Dragonfly can be
used directly in the command line by calling
[`dragonfly-script.py`](bin/dragonfly-script.py)
or be imported in python code via the `maximise_function` function in the main library.
To help get started, we have provided some examples in the
[`examples`](examples) directory.
*If you have pip installed, you will need to clone the repository to access the demos.*

**Via command line**:

To use Dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
The optimisation problem can be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) (recommended) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`examples/synthetic/branin/config.json`](examples/synthetic/branin/config.json) and
[`examples/synthetic/branin/config.pb`](examples/synthetic/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`examples/options_example.txt`](examples/options_example.txt).
We have demonstrated these via some examples below.

We recommend using JSON since we have exhaustively tested JSON format.
If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The first example is on the
[Branin](https://www.sfu.ca/~ssurjano/branin.html) benchmark for global optimisation,
which is defined in [`examples/synthetic/branin/branin.py`](examples/synthetic/branin/branin.py).
The branin demo can be run via following commands.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example.txt
$ dragonfly-script.py --config synthetic/branin/config.pb --options options_files/options_example.txt
```
*Minimisation*:
By default, Dragonfly *maximises* functions. To minimise a function, set the
`max_or_min` flag to `min` in the options file as shown in
[`examples/options_example_for_minimisation.txt`](examples/options_example_for_minimisation.txt)
For example,
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example_for_minimisation.txt
```


*Multi-fidelity Optimisation*:
The multi-fidelity version of the branin demo can be run via following command.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config_mf.json --options options_files/options_example.txt
```

Dragonfly can be run on Euclidean, integral, discrete, and discrete numeric domains, or a
domain which includes a combination of these variables.
See other demos on synthetic functions in the
[`examples/synthetic`](examples/synthetic) directory.
For example, to run the multi-fidelity [`park2_3`](examples/synthetic/park2_3/park2_3_mf.py)
demo, simply run
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/park2_3/config_mf.json --options options_files/options_example.txt
```

*Optimisation on a time budget*:
Dragonfly also allows specifying a time budget for optimisation.
The next demo is a maximum likelihood problem in computational astrophysics,
where we wish to estimate cosmological parameters from Type Ia supernova data.
The demos for Bayesian optimisation and multi-fidelity Bayesian optimisation
can be run via the following commands.
This uses a time budget of 2 hours.

```bash
$ cd examples
$ dragonfly-script.py --config supernova/config.json --options options_files/options_example_realtime.txt
$ dragonfly-script.py --config supernova/config_mf.json --options options_files/options_example_realtime.txt    # For multi-fidelity version
```


*Other methods*:
BO is ideally suited for expensive function evaluations - it aims to find the optimum
in as few evaluations and invests significant computation to do so.
This pays dividends if the evaluations are expensive.
However,
if your function evaluations are cheap, we recommended using DiRect or PDOO for
Euclidean domains, and evolutionary algorithms for non-Euclidean domains.
See example below.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/branin/config.json --options options_files/options_example_pdoo.txt
$ dragonfly-script.py --config synthetic/park2_3/config_mf.json --options options_files/options_example_ea.txt
```
You will notice that they run significantly faster than BO.
However, these methods will perform worse than BO on the supernova problem as evaluations
are expensive.


&nbsp;

**In python code**:

The main APIs for Dragonfly are declared in
[`dragonfly/__init__.py`](dragonfly/__init__.py) and defined in the 
[`dragonfly/apis`](dragonfly/apis) directory.
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
max_val, max_pt, history = maximise_function(lambda x: -func(x), domain, max_capital)
print(max_val, max_pt)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be optimised,
and `max_capital` is the capital available for optimisation.
In simple sequential settings, `max_capital` is simply the maximum number of evaluations
to `func`, but it can also be used to specify an available time budget and other forms
of resource constraints. The `maximise_function` returns history along with the optimum value
and the corresponding optimum point. `history.query_points` contains the points queried by the
algorithm and `history.query_vals` contains the function values.

[`examples/synthetic/branin/in_code_demo.py`](examples/synthetic/branin/in_code_demo.py) and
[`examples/supernova/in_code_demo.py`](examples/supernova/in_code_demo.py)
demonstrate the use case for the branin and supernova problems respectively.
To execute these files, simply run
```bash
$ python examples/synthetic/branin/in_code_demo.py
$ python examples/supernova/in_code_demo.py
```

&nbsp;

**Multi-objective optimisation**

Dragonfly also provides functionality for multi-objective optimisation.
Some synthetic demos are available in the `multiobjective_xxx` directories in
[`demos_synthetic`](demos_synthetic).
For example, to run the
[`hartmann`](examples/synthetic/multiobjective_hartmann/multiobjective_hartmann.py)
demo, simply run
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/multiobjective_hartmann/config.json --options options_files/options_example_moo.txt
```

Similarly, you can import and run this in python code via,
```python
from dragonfly import multiobjective_maximise_functions, multiobjective_minimise_functions
...
pareto_max_values, pareto_points, history = multiobjective_maximise_functions(funcs, domain, max_capital)
pareto_min_values, pareto_points, history = multiobjective_minimise_functions(funcs, domain, max_capital)
```
Here, `funcs` is a list of functions to be maximised,
`domain` is the domain  and `max_capital` is the capital available for optimisation.
`pareto_values` are the
[Pareto optimal](https://en.wikipedia.org/wiki/Multi-objective_optimization#Introduction)
function values and `pareto_points` are the corresponding points in `domain`.

&nbsp;

**Neural Architecture Search**

Dragonfly has APIs for defining neural network architectures, defining distances
between them, and also implements the
[NASBOT](https://arxiv.org/pdf/1802.07191.pdf) algorithm,
which uses Bayesian optimisation and optimal transport for neural architecture search.
You will first need to install the following dependencies.
Cython will also require
a C++ library already installed in the system.
```bash
$ pip install cython POT
```

Below is an architecture search demo on a synthetic function.
See the [`examples/nas`](examples/nas) directory for demos of NASBOT in using some large
datasets.
```bash
$ cd examples
$ dragonfly-script.py --config synthetic/syn_cnn_1/config.json --options options_files/options_example.txt
```


&nbsp;

### Contributors

Kirthevasan Kandasamy: [github](https://github.com/kirthevasank),
[webpage](http://www.cs.cmu.edu/~kkandasa/)  
Karun Raju Vysyaraju: [github](https://github.com/karunraju),
[linkedin](https://www.linkedin.com/in/karunrajuvysyaraju)  
Willie Neiswanger: [github](https://github.com/willieneis),
[webpage](http://www.cs.cmu.edu/~wdn/)  
Biswajit Paria: [github](https://github.com/biswajitsc),
[webpage](https://biswajitsc.github.io/)  
Chris Collins: [github](https://github.com/crcollins/),
[webpage](https://www.crcollins.com/)


### Acknowledgements
Research and development of the methods in this package were funded by
DOE grant DESC0011114, NSF grant IIS1563887, the DARPA D3M program, and AFRL.


### Citation
If you use any part of this code in your work, please cite
[this manuscript](http://www.cs.cmu.edu/~kkandasa/docs/proposal.pdf).

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/blob/master/LICENSE.txt).

For questions, please email kandasamy@cs.cmu.edu.

"Copyright 2018-2019 Kirthevasan Kandasamy"


