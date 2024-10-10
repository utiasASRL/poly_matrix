# PolyMatrix 

[![Python application](https://github.com/utiasASRL/poly_matrix/actions/workflows/python-app.yml/badge.svg)](https://github.com/utiasASRL/poly_matrix/actions/workflows/python-app.yml)

This repository contains functionalities to conveniently set up sparse symmetric matrices arising in semidefinite relaxations of polynomial optimization problems. The core functionality is demonstrated in `example.py`.

For more usage examples, check `_test/test_poly_matrix.py`. 

## Installation

This software was last built using `Python 3.10.15` on `Ubuntu 22.04.1`. If you have any problems with installation, feel free to open an issue. You can install the package locally using
```
conda env create -f environment.yml
conda activate poly_matrix
```

To check that the installation was successful, you can run
```
pytest .
```

To run a simple example, you can run
```
python example.py
```
