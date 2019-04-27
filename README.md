
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
or be imported in python code via the `maximise_function` function in the main library.
To help get started, we have provided some examples in the
[`examples`](examples) directory.
See our readthedocs getting started pages
([command line](https://dragonfly-opt.readthedocs.io/en/master/getting_started_cli/),
[Python](https://dragonfly-opt.readthedocs.io/en/master/getting_started_py/))
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
max_val, max_pt, history = maximise_function(lambda x: -func(x), domain, max_capital)
print(max_val, max_pt)
```
Here, `func` is the function to be maximised,
`domain` is the domain over which `func` is to be optimised,
and `max_capital` is the capital available for optimisation.
The domain can be specified via a JSON file or in code.
See
[here](examples/synthetic/branin/in_code_demo.py),
[here](examples/synthetic/hartmann6_4/in_code_demo.py),
[here](examples/synthetic/discrete_euc/in_code_demo_1.py),
[here](examples/synthetic/discrete_euc/in_code_demo_2.py),
[here](examples/synthetic/hartmann3_constrained/in_code_demo.py),
[here](examples/synthetic/park1_constrained/in_code_demo.py),
[here](examples/synthetic/borehole_constrained/in_code_demo.py),
[here](examples/synthetic/multiobjective_branin_currinexp/in_code_demo.py),
[here](examples/synthetic/multiobjective_hartmann/in_code_demo.py),
[here](examples/tree_reg/in_code_demo.py),
and
[here](examples/nas/demo_nas.py)
for more detailed examples.



For a comprehensive list of uses cases, including multi-objective optimisation,
multi-fidelity optimisation, neural architecture search, and other optimisation
methods (besides Bayesian optimisation), see our readthe docs pages
([command line](https://dragonfly-opt.readthedocs.io/en/master/getting_started_cli/),
[Python](https://dragonfly-opt.readthedocs.io/en/master/getting_started_py/)).


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
[this manuscript](https://arxiv.org/pdf/1903.06694.pdf).

```
@article{kandasamy2019tuning,
    title={{Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian
            Optimisation with Dragonfly}},
    author={Kandasamy, Kirthevasan and Vysyaraju, Karun Raju and Neiswanger,
            Willie and Paria, Biswajit and Collins, Christopher R. and Schneider, Jeff and
            Poczos, Barnabas and Xing, Eric P},
    journal={arXiv preprint arXiv:1903.06694},
    year={2019}
  }
```

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/dragonfly/dragonfly/blob/master/LICENSE.txt).

For questions, please email kandasamy@cs.cmu.edu.

"Copyright 2018-2019 Kirthevasan Kandasamy"


