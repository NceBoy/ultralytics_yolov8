from pathlib import Path
from shutil import copy
import cv2


source_root = Path('../datasets/cycles/')
target_root = Path('../datasets/cycles_target/')


mapping = {
    '0': 0,
    '1': 0,
    '2': 2   
}

seen = set()

for txtfile in source_root.rglob('**/labels/*.txt'):
    txtname = txtfile.name
    if '_jpeg_jpg' in txtname:
        continue
    fileid = txtname.split('_')[0]
    if fileid in seen:
        continue
    else:
        seen.add(fileid)
    mode = txtfile.parent.parent.name  # train val test
    saved_txtfile = target_root / 'train' / 'labels' / f'{mode}-{txtfile.name}'
    saved_imgfile = target_root / 'train' / 'images' /f'{mode}-{txtfile.stem}.jpg'

    source_imgfile = source_root / mode / 'images' / txtfile.with_suffix('.jpg').name

    saved_txtfile.parent.mkdir(parents=True, exist_ok=True)
    with open(saved_txtfile, 'w') as saved_f:
        with open(txtfile) as f:
            for line in f:
                idx = line[0]
                if idx in mapping:
                    new_line = f'{mapping[idx]}{line[1:]}'
                else:
                    continue
                saved_f.write(new_line)
    saved_imgfile.parent.mkdir(parents=True, exist_ok=True)
    if not txtname.startswith('Motor') and 'PNG_jpg' not in txtname and int(fileid) >= 126:
        img = cv2.imread(str(source_imgfile))
        img = img[:, :, ::-1]
        cv2.imwrite(str(saved_imgfile), img)
    else:
        copy(source_imgfile, saved_imgfile)
    

            

