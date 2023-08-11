# Data

This is an example of how the `CoralNet-Toolbox` expects data to be stored, and how they will be downloaded 
from CoralNet. It is recommended to follow this format, though it's not necessary.

```python
output_dir/

    source_id_1/
        images/
        patches/
            class_1/
            class_2/
            class_3/
        model/
            run_1/
                logs/
                weights/
                best_model.h5
                class_mapping.json
        masks/
            plots/
            segs/
            colors/
            color_mapping.json
        sfm/
            project.psx
        annotations.csv
        labelset.csv
        images.csv
        masks.csv
        metadata.csv
        patches.csv
        predictions.csv
        
    source_id_2/
        images/
        patches/
            class_1/
            class_2/
            class_3/
        model/
            run_1/
                logs/
                weights/
                best_model.h5
                class_mapping.json
        masks/
            plots/
            segs/
            colors/
            color_mapping.json
        sfm/
            project.psx
        annotations.csv
        labelset.csv
        images.csv
        masks.csv
        metadata.csv
        patches.csv
        predictions.csv
        
```