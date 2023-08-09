# Data

This is an example of how the `CoralNet-Toolbox` expects data to be stored, and how they will be downloaded 
from CoralNet. It is recommended to follow this format, though it's not necessary.

```python
output_dir/

    source_id_1/
        images/
        predictions/
        patches/
            class_1/
            class_2/
            class_3/
        model/
            run_1/
                logs/
                weights/
        masks/
            visualize/
        annotations.csv
        labelset.csv
        images.csv
        masks.csv
        metadata.csv
        patches.csv
        predictions.csv
        
    source_id_2/
        images/
        predictions/
        patches/
            class_1/
            class_2/
            class_3/
        model/
            run_1/
                logs/
                weights/
        masks/
            visualize/
        annotations.csv
        labelset.csv
        images.csv
        masks.csv
        metadata.csv
        patches.csv
        predictions.csv
        
```