"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "qoc_ag"
VERSION = "0.1"
DEPENDENCIES = [
    "autograd",
    "filelock",
    "h5py",
    "matplotlib",
    "numba",
    "numpy",
    "qutip",
    "scipy",
    "scqubits",
    "pathos"
]
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Yunwei LU"
AUTHOR_EMAIL = "yunweilu2020@u.northwestern.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
      packages=['qoc_ag']
)