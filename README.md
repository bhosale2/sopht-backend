# sopht-backend
Scalable One-stop Platform for Hydroelastic Things (SOPHT) backend

## Installation

Below are steps of how to install `sopht-backend`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.8.0.

```bash
conda create --name sopht-backend-env
conda activate sopht-backend-env
conda install python==3.8
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```