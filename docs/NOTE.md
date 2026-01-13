# NOTE

## 03/01/2026

- Initialized Project 1: Human vs. Non-Human Face Classification.
- Defined dataset strategy and target volumes for binary and multi-class classification.
- Outlined initial TODO steps for data collection and preprocessing.

## 04/01/2026

### Experiment 1: Binary Classification (YOLO11n-cls)
- **Objective**: Human vs. Non-Human face classification.
- **Results**: Achieved **98.24% Top-1 Accuracy** on the held-out test set (836/851 correctly classified).
- **Inference**: Successfully exported model to **ONNX** format. Verified performance using `onnxruntime`, confirming consistent results between PyTorch and ONNX exports.
- **Dataset Split**: Processed 433 human and 418 non-human samples for testing, showing high robustness across both classes.

### Dataset Configuration
For a robust classification performance, the following dataset strategies are recommended:

#### 1. Binary Classification (Ideal Setup)
*   **Target Volume:** ~10,000 images total.
*   **Class Distribution:** 5,000 images of real humans / 5,000 images of non-humans.
*   **Diversity Requirement:** The non-human class should be subdivided into ~1,250 images each for anime, gaming characters, statues, and 3D avatars to prevent bias toward a specific artistic style.

#### 2. Multi-Class Classification (Advanced Setup)
*   **Target Volume:** ~15,000+ images total.
*   **Recommended Classes:** `Human`, `Anime/Cartoon`, `Game_CGI`, `Physical_Arts (Statues)`.
*   **Ideal Number per Class:** 3,000 to 4,000 images per category.
*   **Complexity Note:** Multi-class classification provides higher utility but requires more precise labeling to distinguish between high-fidelity CGI and real humans.

#### 3. Data Split Strategy
*   **Training (70%):** ~7,000 images for model learning.
*   **Validation (15%):** ~1,500 images for hyperparameter tuning.
*   **Test (15%):** ~1,500 images for final performance benchmarking (unseen data).