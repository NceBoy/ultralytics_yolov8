from ultralytics import YOLO

# Load a model
model = YOLO('/home/yehaochen/codebase/ultralytics/runs/detect/origin_base_teehee_md_new/weights/best.pt')  # load an official model

# Validate the model
metrics = model.val(data='/home/yehaochen/datasets/origin_base_teehee_md_old/data.yaml')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category