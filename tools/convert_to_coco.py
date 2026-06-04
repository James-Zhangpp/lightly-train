import os
import json
import random
from collections import defaultdict
import datetime

def generate_coco_format(json_dir, output_dir, split_ratio=0.8):
    """
    Convert labelme-like json annotations to COCO format and split into train/test.
    """
    print(f"Reading JSON files from {json_dir}...")
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 1. First pass: extract all categories and group images by category for stratification
    categories = {}
    category_id = 1
    
    # Map category name to list of json files containing that category
    category_to_files = defaultdict(list)
    file_to_categories = defaultdict(set)
    
    for jf in json_files:
        with open(os.path.join(json_dir, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape['label']
                if label not in categories:
                    categories[label] = category_id
                    category_id += 1
                category_to_files[label].append(jf)
                file_to_categories[jf].add(label)
                
    print(f"Found categories: {categories}")

    # 2. Split dataset according to categories (stratified split approximation)
    train_files = set()
    test_files = set()
    
    # To handle multi-label images simply, we iterate through categories
    # and try to maintain the split ratio for each category.
    for cat, files in category_to_files.items():
        # Remove duplicates
        files = list(set(files))
        random.shuffle(files)
        
        # Determine how many files we need for training for this category
        num_train = int(len(files) * split_ratio)
        
        # Add to train/test sets, ensuring we don't overwrite a previous assignment
        assigned_train = 0
        for f in files:
            if f in train_files or f in test_files:
                continue # Already assigned by another category
                
            if assigned_train < num_train:
                train_files.add(f)
                assigned_train += 1
            else:
                test_files.add(f)

    # Any remaining unassigned files (if any edge cases) go to train
    for f in json_files:
        if f not in train_files and f not in test_files:
            train_files.add(f)

    print(f"Total files: {len(json_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # 3. Define COCO templates
    def create_coco_base():
        return {
            "info": {
                "description": "Custom COCO Dataset",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "date_created": datetime.datetime.now().strftime('%Y/%m/%d')
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in categories.items()]
        }

    train_coco = create_coco_base()
    test_coco = create_coco_base()

    # 4. Second pass: Parse files and populate COCO
    def process_files(file_set, coco_dict):
        ann_id = 1
        for img_id, jf in enumerate(sorted(list(file_set)), start=1):
            with open(os.path.join(json_dir, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add image info
            img_info = {
                "id": img_id,
                "file_name": data.get("imagePath", jf.replace('.json', '.png')),
                "width": data.get("imageWidth", 0),
                "height": data.get("imageHeight", 0)
            }
            coco_dict["images"].append(img_info)
            
            # Add annotations
            for shape in data.get('shapes', []):
                cat_name = shape['label']
                cat_id = categories[cat_name]
                
                # points is [[x, y], [x, y], ...]
                points = shape.get('points', [])
                if not points:
                    continue
                    
                # Flatten points for COCO segmentation
                segmentation = []
                for p in points:
                    segmentation.extend([p[0], p[1]])
                
                # Calculate bounding box [x, y, w, h] from polygon
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                xmin = min(x_coords)
                xmax = max(x_coords)
                ymin = min(y_coords)
                ymax = max(y_coords)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                
                # Calculate area (simplified as bbox area, or you can implement polygon area)
                area = bbox[2] * bbox[3]
                
                ann_info = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": [segmentation],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                }
                coco_dict["annotations"].append(ann_info)
                ann_id += 1

    process_files(train_files, train_coco)
    process_files(test_files, test_coco)

    # 5. Save COCO JSONs
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=4)
        
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_coco, f, indent=4)
        
    print(f"Saved {train_path}")
    print(f"Saved {test_path}")

if __name__ == "__main__":
    # Configure paths
    base_dir = r"d:\ImageSeg\lightly-train\tools\DataSet\FP-data"
    json_dir = os.path.join(base_dir, "labels")
    output_dir = os.path.join(base_dir, "annotations")
    
    # Run conversion
    generate_coco_format(json_dir, output_dir, split_ratio=0.8)
    print("Conversion completed successfully!")
