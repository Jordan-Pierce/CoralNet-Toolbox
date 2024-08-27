import os
from setuptools import setup, find_packages

# Check if requirements.txt exists
assert os.path.exists("requirements.txt"), print("ERROR: Cannot find requirements.txt")

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Setup
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
            "coralnet-toolbox = toolbox:run"
        ]
    },
    package_data={
        'toolbox': ['icons/*']
    },
)