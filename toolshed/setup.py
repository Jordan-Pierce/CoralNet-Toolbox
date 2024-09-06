import os
import sys
import traceback
import subprocess
from setuptools import setup, find_packages


if not os.path.exists("requirements.txt"):
    raise FileNotFoundError("ERROR: Cannot find requirements.txt")

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='coralnet-toolshed',
    version='0.0.1',
    description='Old tools useful for interacting with CoralNet and other CPCe-related downstream tasks.',
    url='https://github.com/Jordan-Pierce/CoralNet-Toolbox',
    author='Jordan Pierce',
    author_email='jordan.pierce@noaa.gov',
    packages=find_packages(),
    install_requires=required_packages,
    python_requires='>=3.7',
    entry_points={
        "console_scripts": [
            "coralnet-toolshed = src:run"
        ]
    },
)

this_dir = os.path.dirname(os.path.realpath(__file__))
whl_path = f'{this_dir}\\Packages\\Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl'
if not os.path.exists(whl_path):
    raise FileNotFoundError("ERROR: Cannot find Metashape wheel file; please install from Metashape website.")

subprocess.check_call([sys.executable, '-m', 'pip', 'install', whl_path],
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE)

print("Metashape installation completed successfully.")