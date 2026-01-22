import os
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_wider_face_annotations(txt_path):
    """
    Parses WIDER FACE annotation file.

    Returns:
        dict: A dictionary where keys are image relative paths and
              values are lists of bounding boxes with attributes.
    """
    annotations = {}
    path = Path(txt_path)
    if not path.exists():
        print(f"Warning: Annotation file {txt_path} not found.")
        return annotations

    with open(txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        file_path = line
        try:
            num_bboxes = int(lines[i + 1].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        bboxes = []
        start_idx = i + 2

        actual_lines_to_skip = num_bboxes
        if num_bboxes == 0:
            if start_idx < len(lines):
                parts = lines[start_idx].strip().split()
                if len(parts) >= 10:
                    actual_lines_to_skip = 1
                else:
                    actual_lines_to_skip = 0
        else:
            for j in range(num_bboxes):
                if (start_idx + j) >= len(lines):
                    break
                parts = lines[start_idx + j].strip().split()
                if len(parts) >= 4:
                    bboxes.append(
                        {
                            "bbox": [
                                int(parts[0]),
                                int(parts[1]),
                                int(parts[2]),
                                int(parts[3]),
                            ],
                            "blur": int(parts[4]) if len(parts) > 4 else 0,
                            "expression": int(parts[5]) if len(parts) > 5 else 0,
                            "illumination": int(parts[6]) if len(parts) > 6 else 0,
                            "invalid": int(parts[7]) if len(parts) > 7 else 0,
                            "occlusion": int(parts[8]) if len(parts) > 8 else 0,
                            "pose": int(parts[9]) if len(parts) > 9 else 0,
                        }
                    )

        annotations[file_path] = bboxes
        i = start_idx + actual_lines_to_skip

    return annotations


def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_widerface(detector, annotations, images_dir, iou_threshold=0.3):
    """
    Evaluates detector performance on WIDER FACE dataset with attribute breakdown.
    """
    results = {
        "total_gt": 0,
        "total_dt": 0,
        "matches": 0,
        "mae_list": [],
        "times_ms": [],
        "attr_recall": {
            "blur": {0: [0, 0], 1: [0, 0], 2: [0, 0]},  # [matches, total]
            "expression": {0: [0, 0], 1: [0, 0]},
            "illumination": {0: [0, 0], 1: [0, 0]},
            "occlusion": {0: [0, 0], 1: [0, 0], 2: [0, 0]},
            "pose": {0: [0, 0], 1: [0, 0]},
            "invalid": {0: [0, 0], 1: [0, 0]},
        },
    }

    images_path = Path(images_dir)

    for rel_path, gt_list in tqdm(annotations.items(), desc="Evaluating WIDER FACE"):
        img_p = images_path / rel_path
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        # Convert WIDER FACE [x1, y1, w, h] to [x1, y1, x2, y2]
        gt_boxes = []
        for ann in gt_list:
            x1, y1, w, h = ann["bbox"]
            gt_boxes.append([x1, y1, x1 + w, y1 + h, ann])

        # Run inference and measure time
        start_time = time.time()
        if hasattr(detector, "detect"):
            detected_faces = detector.detect(img)
        elif hasattr(detector, "get"):
            detected_faces = detector.get(img)
        else:
            raise AttributeError("Model must have a .detect() or .get() method")
        end_time = time.time()

        results["times_ms"].append((end_time - start_time) * 1000)

        # Extract detected boxes [x1, y1, x2, y2]
        dt_boxes = [face.bbox[:4] for face in detected_faces]

        results["total_gt"] += len(gt_boxes)
        results["total_dt"] += len(dt_boxes)
        results["mae_list"].append(abs(len(gt_boxes) - len(dt_boxes)))

        # Update attribute totals
        for gt in gt_boxes:
            attr = gt[4]
            results["attr_recall"]["blur"][attr["blur"]][1] += 1
            results["attr_recall"]["expression"][attr["expression"]][1] += 1
            results["attr_recall"]["illumination"][attr["illumination"]][1] += 1
            results["attr_recall"]["occlusion"][attr["occlusion"]][1] += 1
            results["attr_recall"]["pose"][attr["pose"]][1] += 1
            results["attr_recall"]["invalid"][attr["invalid"]][1] += 1

        # Match detections to ground truth
        matched_gt_indices = set()
        for dt_box in dt_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_data in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue
                iou = calculate_iou(dt_box, gt_data[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                matched_gt_indices.add(best_gt_idx)
                results["matches"] += 1

                # Update attribute matches
                attr = gt_boxes[best_gt_idx][4]
                results["attr_recall"]["blur"][attr["blur"]][0] += 1
                results["attr_recall"]["expression"][attr["expression"]][0] += 1
                results["attr_recall"]["illumination"][attr["illumination"]][0] += 1
                results["attr_recall"]["occlusion"][attr["occlusion"]][0] += 1
                results["attr_recall"]["pose"][attr["pose"]][0] += 1
                results["attr_recall"]["invalid"][attr["invalid"]][0] += 1

    # Final Metrics
    recall = results["matches"] / results["total_gt"] if results["total_gt"] > 0 else 0
    precision = (
        results["matches"] / results["total_dt"] if results["total_dt"] > 0 else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    avg_time = np.mean(results["times_ms"])

    # Format attribute results
    attr_stats = {}
    for attr_name, levels in results["attr_recall"].items():
        attr_stats[attr_name] = {
            level: (f"{matches / total:.2%}" if total > 0 else "N/A")
            for level, (matches, total) in levels.items()
        }

    return {
        "Overall": {
            "MAE": round(np.mean(results["mae_list"]), 2),
            "F1 Score": round(f1, 4),
            "Recall": f"{recall:.2%}",
            "Precision": f"{precision:.2%}",
            "Avg Time (ms)": round(avg_time, 2),
            "Total Images": len(results["mae_list"]),
            "Total GT": results["total_gt"],
            "Total Det": results["total_dt"],
        },
        "Recall per Attribute": attr_stats,
    }


def visualize_wider_face(img_path, bboxes, title="WIDER FACE Visualization"):
    """
    Visualize an image with its bounding boxes.
    """
    path = Path(img_path)
    if not path.exists():
        print(f"Error: Image {img_path} not found.")
        return

    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: Could not read image {img_path}.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    for ann in bboxes:
        x, y, w, h = ann["bbox"]
        rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)

    plt.title(f"{title} ({len(bboxes)} faces)")
    plt.axis("off")
    plt.show()


def get_full_img_path(root_dir, split, rel_path):
    """
    Constructs the full path to an image based on the split.
    """
    root = Path(root_dir)
    split_subdir = {
        "train": "WIDER_train/WIDER_train/images",
        "val": "WIDER_val/WIDER_val/images",
        "test": "WIDER_test/WIDER_test/images",
    }
    return root / split_subdir.get(split, "") / rel_path
