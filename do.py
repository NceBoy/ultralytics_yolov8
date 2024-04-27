import os
from os.path import join
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# Define path to the input image directory (Directory A)
input_dir = '../datasets/origin/images/train'

# Define path to the output directory (Directory B)
output_dir = '../imageshow/'
d_output_dir = '../datasets/origin/labels/train/'
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Run inference on each image in the input directory
for image_file in image_files:
    # Define path to the current image
    image_path = join(input_dir, image_file)
    txt_file = os.path.splitext(image_file)[0] + '.txt'
    txt_path = join(output_dir, txt_file)
    d_txt_path = join(d_output_dir, txt_file)
    # Run inference on the current image
    print(image_path)
    results = model(image_path)
    for result in results:
        result.save_txt(txt_path)
    try:
        with open(txt_path, 'r') as txt_file:
            lines = txt_file.readlines()
        # Filter out lines with class label 0
        class1_lines = [line.strip().replace('0 ', '2 ') + '\n' for line in lines if line.split()[0] == '0']
    except FileNotFoundError:
        print(f"Warning: File not found: {txt_path}")
        continue
    
    # Write the filtered content to the txt file in directory D
    os.makedirs(os.path.dirname(d_txt_path), exist_ok=True)
    with open(d_txt_path, 'a') as d_txt_file:
        d_txt_file.writelines(class1_lines)
    
