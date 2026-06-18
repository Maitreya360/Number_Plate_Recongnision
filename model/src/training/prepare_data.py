import argparse
import os
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def build_dataset(raw_dir, yolo_dir):
    raw_path = Path(raw_dir)
    yolo_path = Path(yolo_dir)
    
    # 1. Create YOLO structure
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_path / p).mkdir(parents=True, exist_ok=True)

    # 2. Generate data.yaml
    yaml_path = yolo_path / "data.yaml"
    if not yaml_path.exists():
        with open(yaml_path, "w") as f:
            f.write(f"path: {yolo_path.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write("nc: 1\n")
            f.write("names: ['license_plate']\n")

    print(f"Scanning '{raw_dir}' for XMLs and Images regardless of folder structure...")
    
    # 3. SMART MAPPER: Find ALL images everywhere and log their paths
    image_map = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    for img_path in raw_path.rglob("*"):
        if img_path.suffix in valid_exts:
            # Save the file name without the extension (the stem)
            image_map[img_path.stem] = img_path
            
    # 4. Find ALL XML files everywhere
    xml_files = list(raw_path.rglob("*.xml"))
    random.shuffle(xml_files)
    
    success_count = 0
    for xml_file in xml_files:
        # Check if we found a matching image somewhere in the giant image_map
        if xml_file.stem not in image_map:
            print(f"Warning: Found '{xml_file.name}' but could not find a matching image anywhere. Skipping.")
            continue
            
        img_path = image_map[xml_file.stem]
        
        folder_type = "train" if random.random() < 0.8 else "val"
        
        target_img_path = yolo_path / "images" / folder_type / img_path.name
        target_lbl_path = yolo_path / "labels" / folder_type / f"{xml_file.stem}.txt"
        
        if target_img_path.exists() and target_lbl_path.exists():
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            with open(target_lbl_path, "w") as out_file:
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_bbox((w, h), b)
                    out_file.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
            
            shutil.copy(img_path, target_img_path)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing '{xml_file.name}': {e}")

    print(f"Successfully paired and added {success_count} new images to the training pile.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    # Changed default to match your exact folder name "test_samples"
    parser.add_argument("--yolo", default=r"data\test_samples") 
    args = parser.parse_args()
    
    build_dataset(args.raw, args.yolo)