from pathlib import Path
from ultralytics import YOLO
import cv2

# Load a model
predictor = YOLO('/home/yehaochen/codebase/ultralytics/runs/detect/origin_base_teehee_md_new/weights/best.pt')  # build a new model from YAML
classify = YOLO('/home/yehaochen/codebase/ultralytics/runs/classify/train6/weights/best.pt')  # build a new model from YAML

# model = model.load('yolov8n-cls.pt')  # build from YAML and transfer weights
save_root = './results'

data_root = Path('/home/yehaochen/datasets/origin_base_teehee_md/images/valid/')

CLS_HUMAN_THRESH = 0.95
CLS_OTHER_THRESH = 0.7

for img in data_root.glob('*.jpg'):
    # Predict
    result = predictor.predict(img)[0]
    bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
    cls_name = [result.names[i] for i in result.boxes.cls.cpu().numpy()]
    pred_conf = result.boxes.conf.cpu().numpy()
    orig_img = result.orig_img
    for conf, pred_cls_name, bbox in zip(pred_conf, cls_name, bboxes):
        x1, y1, x2, y2 = bbox
        crop_img = orig_img[y1:y2, x1:x2]

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.putText(orig_img, pred_cls_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(orig_img, f'{conf:.2f}', (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        if pred_cls_name not in ['ebike', 'bicycle']:
            continue
        
        if conf > 0.7:
            continue

        classify_result = classify.predict(crop_img)[0]
        cls_idx = classify_result.probs.data.argmax().item()
        cls_conf = classify_result.probs.data[cls_idx].item()
        cls_name = classify_result.names[cls_idx]
        if (cls_name != pred_cls_name) and ((cls_name == 'human' and cls_conf > CLS_HUMAN_THRESH) or (cls_name != 'human' and cls_conf > CLS_OTHER_THRESH)):
            # 分类模型容易把 ebike 识别成 human，因此阈值设为较高的 0.9
            cv2.putText(orig_img, cls_name, (x1, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        # for idx, cls_name in enumerate(classify_result.names.values()):
        #     cv2.putText(orig_img, f'{cls_name}: {classify_result.probs.data[idx]:.2f}', (x1, y1 + 60 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)



    cv2.imwrite(f'{save_root}/{img.name}', orig_img)

