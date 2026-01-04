# compvision_learning

Learning Computer Vision

## Project 1: Human vs. Non-Human Face Classification

### Description
This project focuses on developing a deep learning model to differentiate between real human faces and artificial representations. The core objective is to classify whether a detected face belongs to a **Human** or a **Non-Human** entity (encompassing anime, cartoons, video game characters, statues, and digital renders). 

This task is foundational for applications in:
*   **Liveness Detection:** Enhancing security systems against spoofing.
*   **Content Filtering:** Automatically identifying and categorizing artistic vs. real-world content.
*   **Dataset Cleaning:** Sanitizing large-scale face datasets for training biometric systems.

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