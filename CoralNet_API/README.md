# CoralNet API

This repository provides a Python library for accessing data on CoralNet 
through code, which allows users to programmatically interact with CoralNet and 
perform tasks such as uploading and downloading data, and annotating images.

For all of these scripts, it is recommended to set your CoralNet username and 
password as an environment variable, as these will be read as the defaults when 
running the script via command line:
```python
# Windows
set CORALNET_USERNAME=myusername
set CORALNET_PASSWORD=mypassword

# Linux, macOS
export CORALNET_USERNAME=myusername
export CORALNET_PASSWORD=mypassword
```

### CoralNet Download ⬇️

This script can be used to download all data (labelset, images, annotations) 
from any public source, or a private source your account has access to. The 
script is setup to work via command line, and 
expects the following:
- `username` - CoralNet username; will also read the environmental 
  variable `CORALNET_USERNAME`.
- `password` - CoralNet password; will also read the environmental variable 
  `CORALNET_PASSWORD`.
- `source_ids` - A list of source IDs you want to download.
- `output_dir` - A directory that will contain all downloaded data.

Example of use:
```python 
python CoralNet_Download.py --username JohnDoe \
                            --password 123456789 \ 
                            --source_ids 1 54 983 \
                            --output_dir ../CoralNet_Data/
```

If you previously set the environmental variables `CORALNET_USERNAME` and 
`CORALNET_PASSWORD`, these will be read as the defaults, and you can avoid 
passing the respective variables via command line.

Data will be downloaded in the following folder structure (for an example, 
see `../CoralNet_Data`):
```python
output_dir/
    source_id_1/
        images/
        annotations.csv
        labelset.csv
        images.csv
        metadata.csv
    source_id_2/
        images/
        annotations.csv
        labelset.csv
        images.csv
        metadata.csv
```
Although `CoralNet_Download.py` will just download the data for each public 
source desired, it also contains functions that would allow you to:
- Identify all labelsets on CoralNet
- Identify all public sources CoralNet
- Download all public sources given a list of desired labelsets 
- Download all data from all the public sources in CoralNet

Overall, the `CoralNet_Download.py` script is meant to offer researchers 
convenient ways to access and download CoralNet data for further analysis and 
processing. **Please do not abuse CoralNet**: its services are an invaluable 
and useful resource to the coral research community.

### Upload CoralNet ⬆️

This script can be used to automate the process of uploading images, 
annotations, and a labelset given a source ID. The script is setup to work via 
command line, and expects the following:
- `username` - CoralNet username; default env variable `CORALNET_USERNAME`.
- `password` - CoralNet password; default env variable `CORALNET_PASSWORD`.
- `source_id` - The ID of the source you want to upload data to; need access.
- `image_folder` - (optional) A folder with images that you want to upload.
- `annotations` - (optional) A .csv file with point annotations.
- `labelset` - (optional) A .csv file with the labelsets.
- `headless` - (optional) Whether the browser runs in the background

The following describes what is expected in the annotations, and labelset:

```python
# annotations.csv
# image_name, row, column, label are required fields; column names are flexible

                                        image_name   row  column  label
0  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg   671     217   Sand
1  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg  1252     971   Sand
2  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg   548    1054  Macro
```

```python
# labelset.csv
# Label ID and Short Code are requried fields; column names are not flexible

   Label ID     Short_Code 
0        59          Acrop        
1        60          Astreo        
2        61          Cypha        
```

Example of usage:
```python 
python Upload_CoralNet.py --username JohnDoe \
                          --password 123456789 \ 
                          --source_id 1 \
                          --image_folder path/to/images/ \ 
                          --annotations annotations.csv \
                          --labelset labelset.csv
                          --headless True
```

If you previously set the environmental variables `CORALNET_USERNAME` and 
`CORALNET_PASSWORD`, these will be read as the defaults, and you can avoid 
passing the respective variables via command line.

Things to note:
- If you attempt to do the following, CoralNet will throw an error:
  - upload an annotation file without setting any labelsets
  - upload an annotation file with labels that are not within the 
    registered labelset
  - upload an annotation file with images that are not within the source
- The order of uploading should be:
  - labelset and / or images
  - annotations
- If you want to see the uploads occur, set `headless` to False, otherwise 
  it will run in the background (see gif below)

<p align="center">
  <img src="./Figures/CoralNet_Upload.gif" alt="CoralNet_Upload_gif">
  <br>Look Mah, no hands!
</p>


### CoralNet API ☁️

This script can be used to have an existing source's model perform predictions 
on publicly available images. CoralNet's API expects URLs that are 
publicly available, which it will then download and predictions for, given a
list of points. Instead of uploading images to a cloud storage and 
passing those URLs to the API, this script takes advantage of the 
functions in `CoralNet_Dowload.py` by getting the AWS URLs of images 
that have **already uploaded to CoralNet**. Therefore, users must first upload 
the images to the source (w/ or w/o annotations) and `CoralNet_API.py` will 
retrieve the AWS URL given just the `image_name`, and pass it to the model. 
Alternatively, you could use the functions in the script to pass your own URLs.
The script is setup to work via command line, and expects the following:
- `username` - CoralNet username; default env variable `CORALNET_USERNAME`.
- `password` - CoralNet password; default env variable `CORALNET_PASSWORD`.
- `source_id` - The ID of the source you want to upload data to; need access.
- `output_dir` - The directory where you want the predictions to be saved.
- `csv_path` - Path to a .csv file (or folder containing multiple .csv files) 
  with (at a minimum) a column called `image_name`; users can also provide 
  their own points by providing columns `row` and `column` containing 
  points to make predictions for, else, the code will sample the points
  (default is 200 points, stratified random).

The following describes what is expected in the csv files:
```python
# csv_path
# image_name is required; row and column are optional. Column names are flexible

                                        image_name   row  column  
0  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg   671     217   
1  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg  1252     971   
2  mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg   548    1054  
```

If `row` and `column` are not provided, points will be sampled and saved in 
a "points" folder located in the source folder within the output directory. 
Predictions from the model will be saved in a "predictions" folder located
in the source folder within the output directory.

Users can also refer to the API documentation provided by CoralNet for more 
information on the available API endpoints and their usage.


### Notebooks

Before jumping into the scipts, it might be useful to play around with 
functions for each script via notebook. The notebooks contain examples and 
comments that may be useful for understanding and altering the code as 
needed. The notebooks are as follows:
- `CoralNet_Download.ipynb` - Notebook for downloading data from CoralNet.
- `CoralNet_Uplod.ipynb` - Notebook for uploading data to CoralNet.
- `CoralNet_API.ipynb` - Notebook for making predictions via CoralNet's API.

**Pull requests are welcome!**