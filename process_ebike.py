from pathlib import Path
from shutil import copy


source_root = Path('../datasets/ebike/')
target_root = Path('../datasets/ebike_target')


mapping = {
    '2': 1,
    '3': 1,
    '4': 0,
    '5': 0,
    '7': 2   
}


for txtfile in source_root.rglob('**/labels/*.txt'):
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
    copy(source_imgfile, saved_imgfile)

            

