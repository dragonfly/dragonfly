<span style="font-size:3em">**Installation**</span>

&nbsp;

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

&nbsp;

You can now install Dragonfly via one of the four steps below.

**1. Installation via pip (recommended):**
Installing dragonfly properly requires that numpy is already installed in the current
environment. Once that has been done, the library can be installed with pip.

```bash
$ pip install numpy
$ pip install dragonfly-opt -v
```


&nbsp;

**2. Installation via source:**
To install via source, clone the repository and proceed as follows.
```bash
$ git clone https://github.com/dragonfly/dragonfly.git
$ cd dragonfly
$ pip install -r requirements.txt
$ python setup.py install
```

&nbsp;


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

&nbsp;


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
[`direct_fortran`](https://github.com/dragonfly/dragonfly/tree/master/dragonfly/utils/direct_fortran) directory.
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


&nbsp;


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
>>> # The first argument below is the function, the second is the domain, and the third is
>>> # the budget.
>>> min_val, min_pt, history = minimise_function(lambda x: x ** 4 - x**2 + 0.1 * x, [[-10,
>>> 10]], 10);  
...
>>> min_val, min_pt
(-0.32122746026750953, array([-0.7129672]))
```
Due to stochasticity in the algorithms, the above values for `min_val`, `min_pt` may be
different. If you run it for longer (e.g.
`min_val, min_pt, history = minimise_function(lambda x: x ** 4 - x**2 + 0.1 * x, [[-10, 10]], 100)`),
you should get more consistent values for
the minimum.

To run all unit tests, run ```bash run_all_tests.sh```. Some of the tests are
stochastic and could fail occasionally. If this happens, run the same test a few times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simply via
`python -m unittest path.to.test.unittest_xxx`.


