# CoralNet Data

This is an example of how the scripts expect data to be stored, and how they will be downloaded 
from CoralNet. Additionally, if you the CoralNet API and Image Classifier code, they will also 
utilize this folder structure to store predictions and training data, respectively.

```python
output_dir/

    CoralNet_Labelset_Dataframe.csv
    CoralNet_Source_ID_Dataframe.csv
    
    source_id_1/
        images/
        predictions/
        patches/
            train/
            valid/
            test/
        annotations.csv
        labelset.csv
        images.csv
        metadata.csv
        
    source_id_2/
        images/
        predictions/
        patches/
            train/
            valid/
            test/
        annotations.csv
        labelset.csv
        images.csv
        metadata.csv
        
```