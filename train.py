from ultralytics import YOLO
import sys
from pathlib import Path


# Load a model
data_yaml = sys.argv[1]
# for data_yaml in data_yamls:
model = YOLO('yolov8n.pt')
# Train the model with 2 GPUs
results = model.train(data=data_yaml, epochs=100, imgsz=640,  batch=64, name=f'{Path(data_yaml).stem}_aug_ft')