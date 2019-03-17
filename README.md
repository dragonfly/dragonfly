
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

For detailed instructions on installing Dragonfly and its dependencies, see the
[documentation](https://dragonfly-opt.readthedocs.io/en/docs/install/).

**Quick Installation:**
If you have done this kind of thing before, you should be able to install
Dragonfly via `pip`.

```bash
$ sudo apt-get install python-dev python3-dev gfortran # On Ubuntu/Debian
$ pip install numpy
$ pip install dragonfly-opt -v
```
If the installation fails or if there are warning messages, see detailed instructions
[here](https://dragonfly-opt.readthedocs.io/en/docs/install/).



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


