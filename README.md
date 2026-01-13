# compvision_learning

Learning Computer Vision

## Project 1: Human vs. Non-Human Face Classification

### Description
This project focuses on developing a deep learning model to differentiate between real human faces and artificial representations. The core objective is to classify whether a detected face belongs to a **Human** or a **Non-Human** entity (encompassing anime, cartoons, video game characters, statues, and digital renders). 

This task is foundational for applications in:
*   **Liveness Detection:** Enhancing security systems against spoofing.
*   **Content Filtering:** Automatically identifying and categorizing artistic vs. real-world content.
*   **Dataset Cleaning:** Sanitizing large-scale face datasets for training biometric systems.


## Project 2: Face Analysis in Crowds

### Description
Evaluate the efficiency and accuracy of state-of-the-art face detection models (**InsightFace** and **Uniface**) in complex, high-density crowd scenarios.

### Objectives
*   **Crowd Benchmarking:** Assess how well models detect small, partially occluded, or distant faces in crowded environments.
*   **Comparative Analysis:** Performance comparison between `InsightFace` (SCRFD/Buffalo_L) and `Uniface` (YOLOv5Face) models.
*   **Efficiency Metrics:**
    *   **Detection Rate:** Percentage of ground-truth faces correctly detected.
    *   **Count Error:** Mean Absolute Error (MAE) between detected face count and ground truth.
    *   **Precision/Recall:** Statistical evaluation of detection quality.

### Dataset
*   **Source:** [Crowd Counting Dataset (Roboflow)](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w)
*   **Format:** YOLOv5 PyTorch (Normalized bounding boxes).
*   **Split:** Evaluated on the `test` set containing various crowd densities and lighting conditions.