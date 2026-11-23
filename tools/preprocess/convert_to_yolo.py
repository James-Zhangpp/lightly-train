import os
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

def generate_yolo_format(json_dir, image_source_dir, output_root, default_split_ratio=0.8, category_split_ratios=None, category_mapping=None):
    """
    Convert labelme-like json annotations to YOLO segmentation format and organize train/val sets.
    """
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    
    # 1. Extract categories (0-indexed for YOLO)
    categories = category_mapping if category_mapping is not None else {}
    category_id = max(categories.values()) + 1 if categories else 0
    
    category_to_files = defaultdict(list)
    
    for jf in json_files:
        with open(os.path.join(json_dir, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape['label']
                # 如果用户传了固定 mapping 但 label 不在其中，则跳过
                if category_mapping is not None and label not in categories:
                    continue
                    
                if category_mapping is None and label not in categories:
                    categories[label] = category_id
                    category_id += 1
                category_to_files[label].append(jf)
                
    print(f"Extracted Categories: {categories}")
    # Save categories to a reference file so we can update train_inst.py easily
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "class_names.json"), "w", encoding="utf-8") as f:
        # Reverse mapping for train_inst.py
        reversed_cat = {v: k for k, v in categories.items()}
        json.dump(reversed_cat, f, indent=4)

    # 2. Split dataset
    train_files = set()
    val_files = set()
    
    if category_split_ratios is None:
        category_split_ratios = {}

    for cat, files in category_to_files.items():
        files = list(set(files))
        random.shuffle(files)
        
        # 获取当前类别的特定比例，如果没设置则使用默认比例
        ratio = category_split_ratios.get(cat, default_split_ratio)
        num_train = int(len(files) * ratio)
        
        assigned_train = 0
        for f in files:
            if f in train_files or f in val_files:
                continue
                
            if assigned_train < num_train:
                train_files.add(f)
                assigned_train += 1
            else:
                val_files.add(f)

    for f in json_files:
        if f not in train_files and f not in val_files:
            train_files.add(f)

    print(f"Total files: {len(json_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    # 3. Create output directories
    out_dir = Path(output_root)
    for split in ['train', 'val']:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 4. Parse files, convert points, and copy images
    def process_split(file_set, split_name):
        for jf in file_set:
            json_path = os.path.join(json_dir, jf)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_filename = data.get("imagePath", jf.replace('.json', '.png'))
            img_width = data.get("imageWidth", 1)
            img_height = data.get("imageHeight", 1)
            
            # Prepare YOLO txt output
            yolo_lines = []
            for shape in data.get('shapes', []):
                cat_name = shape['label']
                
                # 跳过不在 categories 中的类别
                if cat_name not in categories:
                    continue
                
                cat_id = categories[cat_name]
                
                points = shape.get('points', [])
                if not points:
                    continue
                    
                # Normalize points
                normalized_points = []
                for p in points:
                    x = p[0] / img_width
                    y = p[1] / img_height
                    normalized_points.extend([f"{x:.6f}", f"{y:.6f}"])
                
                yolo_lines.append(f"{cat_id} " + " ".join(normalized_points))
            
            # Write txt
            txt_filename = jf.replace('.json', '.txt')
            txt_path = out_dir / "labels" / split_name / txt_filename
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines) + "\n")
                
            # Copy image
            src_img_path = os.path.join(image_source_dir, img_filename)
            # fallback if image name doesn't match exactly
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(image_source_dir, jf.replace('.json', '.png'))
                img_filename = jf.replace('.json', '.png')
                
            if os.path.exists(src_img_path):
                dst_img_path = out_dir / "images" / split_name / img_filename
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image not found {src_img_path}")

    process_split(train_files, 'train')
    process_split(val_files, 'val')
    print(f"YOLO format dataset created at {output_root}")

if __name__ == "__main__":
    base_dir = r"d:\ImageSeg\lightly-train\tools\DataSet\FP-data"
    json_dir = os.path.join(base_dir, "labels")
    image_dir = os.path.join(base_dir, "images")
    output_root = r"d:\ImageSeg\lightly-train\tools\DataSet\weldingseg-yolo"
    
    # 您可以在这里为特定类别手动设置训练集的比例
    # 未设置的类别将默认使用 default_split_ratio (0.8)
    custom_ratios = {
        "NG": 0.7,  # NG 类别按 7:3 划分
        "OK": 0.8   # OK 类别按 8:2 划分
    }
    
    # 类别名称和 ID 的手动映射（您可以随意更改这里的数字）
    # 如果标注文件中遇到了不在这个字典里的标签，脚本会自动跳过它
    custom_mapping = {
        "ignore": 0,
        "OK": 1,
        "NG": 2
    }
    
    # Run conversion
    generate_yolo_format(
        json_dir, 
        image_dir, 
        output_root, 
        default_split_ratio=0.8, 
        category_split_ratios=custom_ratios,
        category_mapping=custom_mapping
    )
