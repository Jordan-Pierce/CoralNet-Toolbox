import os
import sys
import traceback
import subprocess
from setuptools import setup, find_packages

# Check if requirements.txt exists
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
)


def install_wheel(whl_path):
    try:
        if not os.path.exists(whl_path):
            raise FileNotFoundError(
                f"ERROR: Cannot find Metashape wheel file at {whl_path}; please install from Metashape website.")

        # Install the wheel file
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', whl_path],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

        print("Metashape installation completed successfully.")

    except Exception as e:
        print(f"ERROR: Could not install Metashape.\n{e}\n{traceback.format_exc()}")


try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    whl_path = os.path.join(this_dir, 'Packages', 'Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl')

    # Normalize the wheel path
    whl_path = os.path.normpath(whl_path)

    # Install the wheel file
    install_wheel(whl_path)

except Exception as e:
    print(f"ERROR: An unexpected error occurred.\n{e}\n{traceback.format_exc()}")