# gpbilby

`gpbilby` is a Python package built on top of the **bilby** ecosystem, providing tools for gravitational-wave data analysis and inference, with optional integration of the LIGO Scientific Collaboration (LAL) software stack.

The package is designed to be installable via **pip**, while allowing advanced functionality when LAL is available.

---
### Usage
This package can be used to reproduce the plots and results of the papaer Emma:2026 

## Installation

### Basic installation (recommended)

Install the core package from PyPI:

```bash
pip install gpbilby

###An environment with working dependencies can be created using
conda env create -f requirements.yml

## gpbilby can then be installed by running
pip install gpbilby