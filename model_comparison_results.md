# Model Comparison Results

This project compares the performance of different CNN models implemented for the CIFAR-10 dataset:

## Models Compared

1. **Initial CNN Baseline** - Basic CNN model expecting grayscale input
2. **Improved CNN RMSprop with Dropout** - Enhanced CNN with dropout regularization
3. **Fine-tuned EfficientNet RMS** - Transfer learning model using EfficientNet with RMSprop optimizer
4. **EfficientNet Baseline** - Baseline EfficientNet model

## Performance Metrics

| Model Name | Accuracy | Precision | Recall | F1-Score | Loss | Evaluation Time (s) | Prediction Time per Sample (ms) |
|------------|----------|-----------|--------|----------|------|---------------------|---------------------------------|
| Initial CNN Baseline | 0.6599 | 0.6600 | 0.6599 | 0.6590 | 1.0501 | 1.53 | 1.34 |
| Improved CNN RMSprop with Dropout | 0.6799 | 0.6865 | 0.6799 | 0.6781 | 1.0629 | 2.44 | 1.44 |
| Fine-tuned EfficientNet RMS | 0.8935 | 0.8935 | 0.8935 | 0.8935 | 0.3988 | 40.29 | 3.99 |
| EfficientNet Baseline | 0.9196 | 0.9199 | 0.9196 | 0.9196 | 0.6340 | 38.62 | 3.98 |

## Key Findings

- **Best Accuracy**: EfficientNet Baseline (91.96%)
- **Best Precision**: EfficientNet Baseline (91.99%)
- **Best Recall**: EfficientNet Baseline (91.96%)
- **Best F1-Score**: EfficientNet Baseline (91.96%)
- **Fastest Model**: Initial CNN Baseline (1.34ms per sample)
- **Preprocessing Requirements**: 
  - Initial CNN models expect grayscale input (1 channel) with normalized values [0,1]
  - EfficientNet models (with "with_resize") expect RGB input (3 channels) with values in [0,255] range (internal preprocessing)

## Visualization

The comparison is visualized in `model_performance_comparison.png` which shows:
- Model accuracy comparison
- Model prediction speed comparison

## Computational Performance

Prediction speed was measured by timing how long it takes each model to process 100 sample images from the CIFAR-10 test set. The EfficientNet models achieve significantly higher accuracy (over 89%) but are slower than the grayscale CNN models. The EfficientNet Baseline model achieves the highest accuracy (91.96%) which matches the expected performance. The grayscale CNN models are fastest but have lower accuracy. This demonstrates the trade-off between model complexity, accuracy, and computational efficiency.