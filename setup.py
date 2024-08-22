import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='coralnet-toolbox',
    version='0.0.1',
    description='Tools useful for interacting with CoralNet and other CPCe-related downstream tasks.',
    url='https://github.com/Jordan-Pierce/CoralNet-Toolbox',
    author='Jordan Pierce',
    author_email='jordan.pierce@noaa.gov',
    packages=find_packages(),
    install_requires=required_packages,
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "coralnet-toolbox = coralnet_toolbox:run"
        ]
    },
    package_data={
        'coralnet_toolbox': ['icons/*']
    },
)