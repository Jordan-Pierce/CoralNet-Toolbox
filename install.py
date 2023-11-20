import os
import sys
import shutil
import platform
import subprocess
import urllib.request

# ----------------------------------------------
# OS
# ----------------------------------------------
osused = platform.system()

if osused != 'Windows':
    raise Exception("This install script is only for Windows")

# ----------------------------------------------
# Conda
# ----------------------------------------------
# Need conda to install NVCC if it isn't already
console_output = subprocess.getstatusoutput('conda --version')

# Returned 1; conda not installed
if console_output[0]:
    raise Exception("This install script is only for Windows with Conda already installed")

conda_exe = shutil.which('conda')

# ----------------------------------------------
# Python version
# ----------------------------------------------
python_v = f"{sys.version_info[0]}{sys.version_info[1]}"
python_sub_v = int(sys.version_info[1])

# check python version
if python_sub_v != 8:
    raise Exception(f"Only Python 3.{python_sub_v} is supported.")

# ---------------------------------------------
# MSVC for Windows
# ---------------------------------------------
if osused == 'Windows':

    try:
        print(f"NOTE: Installing msvc-runtime")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'msvc-runtime'])

    except Exception as e:
        print(f"There was an issue installing msvc-runtime\n{e}")
        sys.exit(1)

# ----------------------------------------------
# CUDA Toolkit version
# ----------------------------------------------
try:
    # Command for installing cuda nvcc
    conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-11.8.0", "cuda-toolkit", "-y"]

    # Run the conda command
    print("NOTE: Installing CUDA Toolkit 11.8")
    subprocess.run(conda_command, check=True)

except Exception as e:
    print("ERROR: Could not install CUDA Toolkit")
    sys.exit(1)

# ----------------------------------------------
# Tensorflow & Keras
# ----------------------------------------------
try:
    print("NOTE: Installing Tensorflow 2.10.1")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow==2.10.1'])

except Exception as e:
    print("ERROR: Could not install Tensorflow")
    sys.exit(1)

# ----------------------------------------------
# Pytorch
# ----------------------------------------------
try:

    torch_package = 'torch==2.0.0+cu118'
    torchvision_package = 'torchvision==0.15.1+cu118'
    torch_extra_argument1 = '--extra-index-url'
    torch_extra_argument2 = 'https://download.pytorch.org/whl/cu118'

    # Setting Torch, Torchvision versions
    list_args = [sys.executable, "-m", "pip", "install", torch_package, torchvision_package]
    if torch_extra_argument1 != "":
        list_args.extend([torch_extra_argument1, torch_extra_argument2])

    # Installing Torch, Torchvision
    print("NOTE: Installing Torch 2.0.0")
    subprocess.check_call(list_args)

except Exception as e:
    print("ERROR: Could not install Pytorch")
    sys.exit(1)

# ----------------------------------------------
# Other dependencies
# ----------------------------------------------
install_requires = [
    'wheel',

    'numpy',
    'pandas',
    'scipy',
    'scikit_learn',
    'matplotlib',

    'Pillow',
    'opencv_python',
    'scikit_image',
    'albumentations',
    'imgaug',
    'plyfile',

    'Requests',
    'beautifulsoup4',

    'selenium',
    'webdriver_manager',

    'plot_keras_history',
    'segment_anything',
    'segmentation_models_pytorch',

    './Packages/Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl',

    'gooey',
    'gradio',
    'rich',
    'tqdm'
]

# Installing all the other packages
for package in install_requires:

    try:
        print(f"NOTE: Installing {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    except Exception as e:
        print(f"There was an issue installing {package}\n{e}\n")
        print(f"If you're not already, please try using a conda environment with python 3.8")
        sys.exit(1)

# ----------------------------------------------
# Model Weights
# ----------------------------------------------
print('Downloading networks...')
THIS_DIRECTORY = os.path.abspath(__file__)

# Make the Data directory
SAM_DIR = f"{os.path.dirname(THIS_DIRECTORY)}\\Data\\Cache\\SAM_Weights"
os.makedirs(SAM_DIR, exist_ok=True)

# ---------------
# SAM Weights
# ---------------
base_url = "https://dl.fbaipublicfiles.com/segment_anything/"
net_file_names = ["sam_vit_b_01ec64.pth",
                  "sam_vit_l_0b3195.pth",
                  "sam_vit_h_4b8939.pth"]

for net_name in net_file_names:
    path_dextr = f"{SAM_DIR}\\{net_name}"
    if not os.path.exists(path_dextr):
        try:
            url_dextr = base_url + net_name
            print('Downloading ' + url_dextr + '...')
            # Send an HTTP GET request to the URL
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url_dextr, path_dextr)
            print(f"NOTE: Downloaded file successfully")
            print(f"NOTE: Saved file to {path_dextr}")
        except:
            raise Exception("Cannot download " + net_name + ".")

    else:
        print(net_name + ' already exists.')

print("Done.")