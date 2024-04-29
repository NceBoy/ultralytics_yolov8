data_yamls=(\
    # /root/ultralytics_yolov8/ultralytics/cfg/datasets/ebike.yaml \
    /root/ultralytics_yolov8/ultralytics/cfg/datasets/cycles.yaml \
    /root/ultralytics_yolov8/ultralytics/cfg/datasets/teehee.yaml \
    /root/ultralytics_yolov8/ultralytics/cfg/datasets/teehee2.yaml\
)
for yaml in "${data_yamls[@]}"; do
    python train.py $yaml
done
