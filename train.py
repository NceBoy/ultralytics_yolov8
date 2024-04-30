from ultralytics import YOLO
import sys
from pathlib import Path


# Load a model
data_yaml = '/home/yehaochen/datasets/origin_base_teehee_md/data.yaml'
# for data_yaml in data_yamls:
model = YOLO('yolov8n.pt')
# Train the model with 2 GPUs
results = model.train(data=data_yaml, epochs=100, imgsz=640,  batch=64, name='origin_base_teehee_md_w_aug', device=list(range(8)), cls_names={0: 'ebike'})
# results = model.train(data=data_yaml, epochs=100, imgsz=640,  batch=640, name='origin_base_teehee_md', cls_names={0: 'ebike'})