import os
from setuptools import setup, find_packages

# Check if requirements.txt exists
assert os.path.exists("requirements.txt"), "ERROR: Cannot find requirements.txt"

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Filter out any empty lines or comments
required_packages = [line for line in required_packages if line and not line.startswith('#')]
required_packages = [line for line in required_packages if not line.startswith("git+")]

# Setup
setup(
    name='coralnet-toolbox',
    version='0.0.1',
    description='Tools useful for interacting with CoralNet and other CPCe-related downstream tasks.',
    url='https://github.com/Jordan-Pierce/CoralNet-Toolbox',
    author='Jordan Pierce',
    author_email='jordan.pierce@noaa.gov',
    packages=find_packages(include=['toolbox', 'toolbox.*']),
    install_requires=required_packages + [
        "mobile-sam @ git+https://git@github.com/ChaoningZhang/MobileSAM.git",
        "segment-anything @ git+https://git@github.com/facebookresearch/segment-anything.git",
        "sam-2 @ git+https://git@github.com/facebookresearch/sam2.git"
    ],
    python_requires='>=3.10, <3.11',
    entry_points={
        "console_scripts": [
            "coralnet-toolbox = toolbox:run"
        ]
    },
    package_data={
        'toolbox': ['icons/*']
    },
)