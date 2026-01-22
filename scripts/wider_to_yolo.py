import os
import cv2
from pathlib import Path
from tqdm import tqdm


def convert_to_yolo(x1, y1, w, h, img_w, img_h):
    """Converts WIDER FACE [x1, y1, w, h] to YOLO [cx, cy, w, h] normalized."""
    # Adjust to stay within bounds
    x2 = x1 + w
    y2 = y1 + h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Recalculate width/height after clipping
    w = x2 - x1
    h = y2 - y1

    cx = x1 + w / 2
    cy = y1 + h / 2

    return [cx / img_w, cy / img_h, w / img_w, h / img_h]


def parse_wider_face_to_yolo(anno_file, src_img_dir, target_root, split_name):
    """
    Parses WIDER FACE txt and saves individual YOLO label files.
    """
    img_dir = Path(src_img_dir)
    target_img_dir = Path(target_root) / split_name / "images"
    target_lbl_dir = Path(target_root) / split_name / "labels"

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(anno_file, "r") as f:
        lines = f.readlines()

    i = 0
    count = 0
    while i < len(lines):
        rel_path = lines[i].strip()
        if not rel_path:
            i += 1
            continue

        num_bboxes = int(lines[i + 1].strip())

        # Load image once to get dimensions
        img_full_path = img_dir / rel_path
        img = cv2.imread(str(img_full_path))
        if img is None:
            print(f"Error loading {img_full_path}")
            i += 2 + max(1, num_bboxes)
            continue

        h, w = img.shape[:2]

        yolo_lines = []
        start_idx = i + 2

        # Skip count for data lines
        skip = num_bboxes
        if num_bboxes == 0:
            if start_idx < len(lines) and len(lines[start_idx].split()) >= 10:
                skip = 1
            else:
                skip = 0
        else:
            for j in range(num_bboxes):
                parts = [int(val) for val in lines[start_idx + j].strip().split()]
                # x1, y1, w, h are indices 0, 1, 2, 3
                # Check for 'invalid' flag (index 7). 0 = valid, 1 = invalid.
                # Many people skip invalid faces for YOLO training.
                is_invalid = parts[7] if len(parts) > 7 else 0
                if is_invalid == 1:
                    continue

                yolo_box = convert_to_yolo(parts[0], parts[1], parts[2], parts[3], w, h)
                # Format: <class_id> <cx> <cy> <w> <html>
                # Face class is 0
                line = f"0 {' '.join([f'{val:.6f}' for val in yolo_box])}\n"
                yolo_lines.append(line)

        # Save label file (maintain subdir structure or flatten?)
        # YOLO usually prefers a flat or mirroring structure. We will mirror.
        label_rel_path = Path(rel_path).with_suffix(".txt")
        label_out_path = target_lbl_dir / label_rel_path
        label_out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_out_path, "w") as lf:
            lf.writelines(yolo_lines)

        # Optional: Copy image or symlink to have a standard YOLO structure
        # (For this example, we'll assume you point YOLO to the original image dir)
        # But for strictly following YOLO, we'd copy/symlink them.

        i = start_idx + skip
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images in {split_name}...")

    print(f"Finished {split_name}. Total images: {count}")


if __name__ == "__main__":
    ROOT = "<data_dir>/data/widerface_dataset"
    TARGET = "<data_dir>/data/wider_yolo"

    # Define splits
    tasks = [
        {
            "anno": f"{ROOT}/wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt",
            "imgs": f"{ROOT}/WIDER_train/WIDER_train/images",
            "name": "train",
        },
        {
            "anno": f"{ROOT}/wider_face_annotations/wider_face_split/wider_face_val_bbx_gt.txt",
            "imgs": f"{ROOT}/WIDER_val/WIDER_val/images",
            "name": "val",
        },
    ]

    for t in tasks:
        parse_wider_face_to_yolo(t["anno"], t["imgs"], TARGET, t["name"])


"""
# data.yaml
path: <data_dir>/data # root dir
train: widerface_dataset/WIDER_train/WIDER_train/images # path to images
val: widerface_dataset/WIDER_val/WIDER_val/images

# The labels must be in a folder named 'labels' at the same level as 'images'
# OR you can specify the label directory structure.
# Since our labels are in 'wider_yolo/train/labels', 
# it is best to symlink or move them to match the image structure.

names:
  0: face
"""
