from ultralytics import YOLO

# Hardcoded constants
MODEL_PATH = "yolov8n.pt"
DATA_PATH = "path/to/your/data.yaml"
EPOCHS = 10
ITERATIONS = 300


model = YOLO(MODEL_PATH) 

results = model.tune(data=DATA_PATH, 
                     epochs=10, 
                     iterations=300, 
                     optimizer="AdamW", 
                     plots=False, 
                     save=False, 
                     val=False)

print(f"Results: {results}")

