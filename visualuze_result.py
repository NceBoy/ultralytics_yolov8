from pathlib import Path
import cv2
import random

vis_num = 60

image_root = Path('../datasets/cycles_target/train/images')
txt_root = Path('../datasets/cycles_target/train/labels')
vis_root = f'./visroot/{image_root.parent.parent.name}'
Path(vis_root).mkdir(parents=True, exist_ok=True)


label_mapping = {
    '0': 'ebike',
    '1': 'bike',
    '2': '血鸽'
}

vis_targets = random.sample(list(txt_root.glob('*.txt')), k=vis_num)
# vis_targets = [Path('/root/datasets/cycles_target/train/labels/test-366_jpg.rf.2dadd02877a571fe46e2f7b1baacca85.txt')]

for idx, txt in enumerate(vis_targets):
    if idx >= vis_num:
        break
    imgfile = image_root / f'{txt.stem}.jpg'
    img = cv2.imread(str(imgfile))
    img_w = img.shape[1]
    img_h = img.shape[0]
    for line in open(str(txt)):
        label = label_mapping[line[0]]
        cx, cy, w, h = line[1:].split()
        cx = float(cx) * img_w
        cy = float(cy) * img_h
        w = float(w) * img_w
        h = float(h) * img_h

        x1 = int(cx - w / 2)
        x2 = int(cx + w / 2)
        y1 = int(cy - h / 2)
        y2 = int(cy + h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), [255, 0, 0])
        cv2.putText(img, label, (x1, y1), 1, 1, [255, 0, 0])
    cv2.imwrite(f'{vis_root}/{imgfile.name}', img)

    

