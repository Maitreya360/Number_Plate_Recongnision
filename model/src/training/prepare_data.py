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
    
    # Create the YOLO directory structure
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_path / p).mkdir(parents=True, exist_ok=True)

    # Automatically generate the data.yaml file if it doesn't exist
    yaml_path = yolo_path / "data.yaml"
    if not yaml_path.exists():
        with open(yaml_path, "w") as f:
            f.write(f"path: {yolo_path.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write("nc: 1\n")
            f.write("names: ['license_plate']\n")

    # Find all XML files recursively in your raw folder
    xml_files = list(raw_path.rglob("*.xml"))
    random.shuffle(xml_files)
    
    success_count = 0
    for xml_file in xml_files:
        # Look for the matching JPG image
        jpg_file = xml_file.with_suffix('.jpg')
        if not jpg_file.exists():
            # Sometimes extensions are uppercase
            jpg_file = xml_file.with_suffix('.JPG')
            if not jpg_file.exists():
                continue
        
        # Decide if this goes to 'train' (80%) or 'val' (20%)
        folder_type = "train" if random.random() < 0.8 else "val"
        
        target_img_path = yolo_path / "images" / folder_type / jpg_file.name
        target_lbl_path = yolo_path / "labels" / folder_type / xml_file.with_suffix('.txt').name
        
        # Skip if we already added this file in a previous run
        if target_img_path.exists() and target_lbl_path.exists():
            continue

        # Convert XML to YOLO TXT
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
            
            # Copy Image
            shutil.copy(jpg_file, target_img_path)
            success_count += 1
            
        except Exception as e:
            print(f"Skipping {xml_file.name} due to error: {e}")

    print(f"✅ Successfully processed and added {success_count} new images to {yolo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Folder containing new raw XML and JPG files")
    parser.add_argument("--yolo", default="data/test_sample", help="Where to store the training data")
    args = parser.parse_args()
    
    build_dataset(args.raw, args.yolo)