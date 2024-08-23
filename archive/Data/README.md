# Data

This is an example of how the `CoralNet-Toolbox` expects data to be stored, and how they will be downloaded 
from CoralNet. It is recommended to follow this format, though it's not necessary.

```python
output_dir/

    ...    

    source_id_N/
        # Official downloads from CN
        source_id_N_annotations.csv
        source_id_N_labelset.csv
        source_id_N_images.csv
        source_id_N_metadata.csv
    
        images/
            image_1.png
            image_2.png
            ...
            
        # Made locally
        annotations/
            timestamp_annotations.csv
            ...
        
        # Made locally
        points/
            timestamp_points.csv
            ...
            
        # Made locally
        patches/
            timestamp/
                patches.csv
                patches/
                    class_1/
                    class_2/
                    ...
        
        # Made locally            
        classifier/
            timestamp/
                logs/
                weights/
                best_model.h5
                class_mapping.json
        
        # Made locally, or from CN
        predictions/
            classifier_timestamp_prediction.csv
            coralnet_timestamp_predictions.csv
        
        # Made locally
        masks/
            timestamp/
                masks.csv
                plots/
                segs/
                colors/
                color_mapping.json
                
        # Made locally
        sfm/
            timestamp/
                project.psx
                [data products]
                
    ...

        
```