from ultralytics import YOLO

# Load a model
model = YOLO('/root/ultralytics_yolov8/runs/detect/ebike_aug_ft/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# model.predict('../datasets/origin_base/images/valid', save=True, name='ebike_ft_predict')
model.predict('../datasets/origin_base/images/valid/', save=True, name='ebike_aug_ft_predict')


