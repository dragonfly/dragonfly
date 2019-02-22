## Installation

**Installation via pip (recommended):**
Installing dragonfly properly requires that numpy is already installed in the current
environment. Once that has been done, the library can be installed with pip.

```bash
$ pip install numpy
$ pip install git+https://github.com/dragonfly/dragonfly.git
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
```bash
$ virtualenv env
$ source env/bin/activate
(env)$ pip install numpy
(env)$ pip install git+https://github.com/dragonfly/dragonfly.git
```
You can similarly install via source by creating/sourcing the virtualenv and following the
steps above.


**Requirements:**
Dragonfly requires standard python packages such as `numpy`, `scipy`, `future`, and
`six` (for python2/3 compatibility). They should be automatically installed if you follow
the above installation procedure(s). You can also manually install them by
`$ pip install <package-name>`.


**Testing the Installation**:
You can import Dragonfly in python to test if it was installed properly.
If you have installed via source, make sure that you move to a different directory
(e.g. `cd ..`)  to avoid naming conflicts.
```bash
$ python
>>> from dragonfly import minimise_function
>>> min_val, min_pt, history = minimise_function(lambda x: x ** 4 - x**2 + 0.1 * x, [[-10, 10]], 10);
...
>>> min_val, min_pt
(-0.32122746026750953, array([-0.7129672]))
```
To run all unit tests, run ```bash run_all_tests.sh```. Some of the tests are
stochastic and could fail occasionally. If this happens, run the same test a few times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simpy via `python -m unittest path.to.test.unittest_xxx`.
