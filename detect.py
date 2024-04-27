from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
model.predict('../datasets/cycles_target/train/images/test-366_jpg.rf.2dadd02877a571fe46e2f7b1baacca85.jpg', save=True)
