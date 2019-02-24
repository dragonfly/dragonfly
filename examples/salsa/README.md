
An example on the Shrunk Additive Least Squares Approximations method (SALSA) for high
dimensional regression.

To run this demo, you will need to download the `energy_appliances.p` dataset 
in this directory. The dataset is available
[here](http://www.cs.cmu.edu/~kkandasa/dragonfly_datasets.html).

Then, from this directory, you can run,
```bash
$ cd examples
$ dragonfly-script.py --config config_salsa_energy.json --options ../options_files/options_example_realtime.txt
$ dragonfly-script.py --config config_salsa_energy_mf.json --options ../options_files/options_example_realtime.txt # For multi-fidelity version
```

If you use this experiment, please cite the following paper.
  - Kandasamy K, Yu Y, "Additive Approximations in High Dimensional Nonparametric
    Regression via the SALSA", International Conference on Machine Learning, 2016.
  - (Dataset): Candanedo L M, Feldheim V, and Deramaix D, "Data Driven
    Prediction Models of Energy Use of Appliances in a Low-energy House", Energy and
    Buildings, 2017

