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