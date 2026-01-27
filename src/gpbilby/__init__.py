__author__ = "Gregory Ashton"
__email__ = "gregory.ashton@ligo.org"

from .cl import main

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'
