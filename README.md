
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

Dragonfly is compatible with python2 and python3 and has been tested on Linux,
macOS, and Windows platforms.
For questions and bug reports please email kandasamy@cs.cmu.edu.


&nbsp;

## Installation

**Set up:**
We recommend installation via `pip`.
In most Linux environments, it can be installed via one of the commands below,
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


&nbsp;

**Installation via pip (recommended):**
Installing dragonfly properly requires that numpy is already installed in the current
environment. Once that has been done, the library can be installed with pip.

```bash
$ pip install numpy
$ pip install git+https://github.com/dragonfly/dragonfly.git -v
```

**Installation via source:**
To install via source, clone the repository and proceed as follows.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
$ cd dragonfly
$ pip install -r requirements.txt
$ python setup.py install
```


**Installing in a Python Virtual Environment:**
Dragonfly can be pip installed in a python virtualenv.
In Python2, you can follow the steps below.
In Python3, replace  `$ virtualenv env` with `$ python3 -m venv env`.
You can similarly install via source by creating/sourcing the virtualenv and following the
steps above.
```bash
$ virtualenv env
$ source env/bin/activate
(env)$ pip install numpy
(env)$ pip install git+https://github.com/dragonfly/dragonfly.git
```


**Requirements:**
Dragonfly requires standard python packages such as `numpy`, `scipy`, `future`, and
`six` (for Python2/3 compatibility). They should be automatically installed if you follow
the above installation procedure(s). You can also manually install them by
`$ pip install <package-name>`.


**Testing the Installation**:
You can import Dragonfly in python to test if it was installed properly.
If you have installed via source, make sure that you move to a different directory 
(e.g. `cd ..`)  to avoid naming conflicts.
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
To help get started, we have provided some demos in the `demos_synthetic` directory.
If you have pip installed, you will need to clone the repository to access the demos.

**Via command line**:

To use Dragonfly via the command line, we need to specify the optimisation problem (i.e.
the function to be maximised and the domain) and the optimisation parameters.
We have demonstrated these on the
[`branin`](https://www.sfu.ca/~ssurjano/branin.html) function and a
[`face recognition`](http://scikit-learn.org/0.15/auto_examples/applications/face_recognition.html)
example from [`scikit`](http://scikit-learn.org/0.15/index.html) in the [`demos_synthetic`](demos_synthetic) directory.
The former is a common benchmark for global optimisation while the latter is a
model selection problem in machine learning.
The functions to be maximised in these problems are defined in
[`demos_synthetic/branin/branin.py`](demos_synthetic/branin/branin.py) and
[`demos_real/face_rec/face_rec.py`](demos_real/face_rec/face_rec.py) respectively.
The name of this file and the domain should be specified in
[JSON](https://en.wikipedia.org/wiki/JSON) (recommended) or
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format.
See
[`demos_synthetic/branin/config.json`](demos_synthetic/branin/config.json) and
[`demos_synthetic/branin/config.pb`](demos_synthetic/branin/config.pb) for examples.
Then, specify the optimisation parameters in an options file, in the format shown in
[`demos_synthetic/options_example.txt`](demos_synthetic/options_example.txt).

We recommend using JSON since we have exhaustively tested JSON format.
If using protocol buffers, you might need to install this package via
`pip install protobuf`.

The branin demo can be run via following commands.
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example.txt
$ dragonfly-script.py --config demos_synthetic/branin/config.pb --options demos_synthetic/options_example.txt
```
By default, Dragonfly *maximises* functions. To minimise a function, set the
`max_or_min` flag to `min` in the options file as shown in
[`demos_synthetic/options_example_for_minimisation.txt`](demos_synthetic/options_example_for_minimisation.txt)
For example,
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config.json --options demos_synthetic/options_example_for_minimisation.txt
```


The multi-fidelity version of the branin demo can be run via following command.
```bash
$ dragonfly-script.py --config demos_synthetic/branin/config_mf.json --options demos_synthetic/options_example.txt
```

&nbsp;

Dragonfly can be run on Euclidean, integral, discrete, and discrete numeric domains, or a
domain which includes a combination of these variables.
See other demos on synthetic functions in the
[`demos_synthetic`](demos_synthetic) directory.
For example, to run the multi-fidelity [`park2_3`](demos_synthetic/park2_3/park2_3_mf.py)
demo, simply run
```bash
$ dragonfly-script.py --config demos_synthetic/park2_3/config_mf.json --options demos_synthetic/options_example.txt
```


The face recognition demo tunes hyper-parameters of a face regognition model.
It can be run via the following commands.
You will need to install
[`scikit-learn`](http://scikit-learn.org), which can be done via
`pip install scikit-learn`.
Running this demo the first time will be slow since the dataset needs to be downloaded.

```bash
$ dragonfly-script.py --config demos_real/face_rec/config.json --options demos_real/face_rec/options.txt
$ dragonfly-script.py --config demos_real/face_rec/config.pb --options demos_real/face_rec/options.txt
```

&nbsp;

**In python code**:

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

[`demos_synthetic/branin/in_code_demo.py`](demos_synthetic/branin/in_code_demo.py) and
[`demos_real/face_rec/in_code_demo.py`](demos_real/face_rec/in_code_demo.py)
demonstrate the use case for the branin function and face recognition demos respectively.
To execute this file, simply run
```bash
$ python demos_synthetic/branin/in_code_demo.py
$ python demos_real/face_rec/in_code_demo.py
```

&nbsp;

**Multiobjective optimisation**

Dragonfly also provides functionality for multi-objective optimisation.
Some synthetic demos are available in the `multiobjective_xxx` directories in
[`demos_synthetic`](demos_synthetic).
For example, to run the
[`hartmann`](demos_synthetic/multiobjective_hartmann/multiobjective_hartmann.py)
demo, simply run
```bash
$ dragonfly-script.py --config demos_synthetic/multiobjective_hartmann/config.json --options demos_synthetic/multiobjective_options_example.txt
```

Similarly, you can import and run this in python code via,
```python
from dragonfly import multiobjective_maximise_functions
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
Chris Collins helped make this package pip installable.


### Citation
If you use any part of this code in your work, please cite
[this manuscript](http://www.cs.cmu.edu/~kkandasa/docs/proposal.pdf).

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/blob/master/LICENSE.txt).

For questions, please email kandasamy@cs.cmu.edu

"Copyright 2018-2019 Kirthevasan Kandasamy"


