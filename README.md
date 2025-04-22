## ML_Project: Disease Detection from Chest X-Rays using Deep CNN

This project implements a deep convolutional neural network (CNN) for multi-label image classification of 15 thoracic diseases using the NIH Chest X-Ray dataset. It demonstrates the full machine learning pipeline: data preprocessing, model design, fine-tuning, and evaluation, all aimed at building a clinical-grade AI for radiographic diagnosis.

## Highlights

- EfficientNetB0 transfer learning
- Focal Loss to handle class imbalance
- Stratified data splitting with `MultilabelStratifiedShuffleSplit`
- Two-phase training with learning rate scheduling & early stopping
- Per-class threshold optimization for F1-score boosting


## Evaluation Metrics

| Metric    | Global (0.3) | Per-Class Tuned |
|-----------|--------------|-----------------|
| Precision | 0.159        | 0.187           |
| Recall    | 0.393        | 0.358           |
| F1 Score  | 0.224        | 0.236           |
| AUC       | 0.748 → 0.766 (after tuning) |


## Tools & Libraries

- Tensorflow / Keras
- EfficientNet
- `scikit-learn`, `pandas`, `matplotlib`
- Google Colab + Kaggle CLI


## Files

- `ML_Project.ipynb` — Complete notebook with training, evaluation, and plots
- `README.md` — This summary


## Dataset

- NIH Chest X-ray14 via Kaggle  
  [Dataset Link](https://www.kaggle.com/datasets/nih-chest-xrays/data)

