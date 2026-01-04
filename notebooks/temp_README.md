
---
license: mit
task_categories:
- image-classification
tags:
- human-detection
- face-classification
- computer-vision
size_categories:
- 1K<n<10K
---

# Human vs Non-Human Face Dataset

A robust dataset for binary classification between real human faces and non-human face-like objects (statues, art, gaming, anime).

## 📊 Dataset Statistics
| Split | Human | Non-Human | Total |
| :--- | :--- | :--- | :--- |
| **Train** | 3,024 | 2,949 | 5,973 |
| **Validation** | 864 | 842 | 1,706 |
| **Test** | 433 | 422 | 855 |
| **Total** | | | **8,534** |

## 📁 Format
- Images are decodable as **PIL.Image** objects.
- Labels: `0: human`, `1: non_human`.

## 🚀 Quick Start   
```python
from datasets import load_dataset
ds = load_dataset("8Opt/human-nonhuman-face-classification")

# Access test set
example = ds['test'][0]
img, label = example['image'], example['label']
img.show()
