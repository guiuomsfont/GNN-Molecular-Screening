from distutils.core import setup
import os
import subprocess

setup(
    include_package_data=True,
    install_requires=[
        "torch==1.10.2",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "numpy>=1.21.0",
        "rdkit-pypi>=2021.9.5.1",
        "ase>=3.21.0",
        "scikit-learn>=1.0.0",
        "tabulate>=0.8.0",
    ],
    scripts=[
        "src/GNN_DTI/train_test/train.py",
    ],
)
