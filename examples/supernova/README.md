
An example on maximum likelihood estimation using Type Ia supernova data.

You can run this via,
```bash
$ python in_code_demo.py
```
Or you can run this demo from the command line using the following commands.
```bash
$ dragonfly-script.py --config config.json --options options_supernova.txt
$ dragonfly-script.py --config config_mf.json --options options_supernova.txt # For multi-fidelity version
```

&nbsp;


References: If you use any part of this implementation, please cite the following papers.
  - Kandasamy K, Dasarathy G, Schneider J, Poczos B. "Multi-fidelity Bayesian Optimisation
    with Continuous Approximations", International Conference on Machine Learning 2017
  - (dataset) Davis TM et al. "Scrutinizing Exotic Cosmological Models Using ESSENCE
    Supernova Data Combined with Other Cosmological Probes", Astrophysical Journal 2007
  - (likelihood approximation 1) Robertson H. P. "An Intepretation of Page's ``New
    Relativity``", 1936.
  - (likelihood approximation 2) Shchigolev V K. "Calculating Luminosity Distance versus
    Redshift in FLRW Cosmology via Homotopy Perturbation Method", 2016.

