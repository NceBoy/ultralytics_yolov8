from ultralytics import YOLO

# Load a model
model = YOLO('/root/ultralytics_yolov8/runs/detect/ebike_ft/weights/best.pt')  # load an official model

# Validate the model
metrics = model.val(data='/root/ultralytics_yolov8/ultralytics/cfg/datasets/ebike.yaml')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category