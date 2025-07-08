# Pneumonia Detection using InceptionV3 (Transfer Learning)

This project fine-tunes an InceptionV3 model to classify chest X-ray images as either **Pneumonia** or **Normal** using the [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist) dataset provided as a `.npz` file.

---

##  Dataset

- Source: Kaggle - [rijulshr/pneumoniamnist](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)
- Format: `pneumoniamnist.npz` file with:
  - `train_images`, `train_labels`
  - `val_images`, `val_labels`
  - `test_images`, `test_labels`
- Images are grayscale 28x28.


---

##  Model Architecture

- Base: **InceptionV3** pretrained on ImageNet
- Modifications:
  - Input resized to 299×299 RGB
  - `GlobalAveragePooling2D`
  - `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')` for binary classification

---

##  Preprocessing

- Grayscale (28×28) images expanded to 3 channels (RGB)
- Resized to 299×299 using TensorFlow image processing
- Normalized to [0, 1] range

---

##  Training Setup

- Optimizer: `Adam (lr=1e-4)`
- Loss: `Binary Crossentropy`
- Batch Size: `32`
- Epochs: `25`
- EarlyStopping + Dropout for regularization

---

##  Results (Sample Run)

| Metric         | Normal | Pneumonia |
|----------------|--------|-----------|
| Precision      | 0.97   | 0.73      |
| Recall         | 0.40   | 0.99      |
| F1-Score       | 0.57   | 0.84      |
| Support        | 234    | 390       |

>  **Accuracy**: 77.00%  
>  **Macro Avg F1-Score**: 71.00%  
>  **Weighted Avg F1-Score**: 74.00%

> **Insights**: The model is highly sensitive to pneumonia (Recall = 0.99), minimizing false negatives. However, many normal cases are misclassified as pneumonia (Recall = 0.40), which may lead to unnecessary follow-up tests.

---

## Saving the Model

You can save the model after training using:

```python
model.save("inception_pneumonia_model.h5")
```

Or use callbacks to save the best weights:

```python
from tensorflow.keras.callbacks import ModelCheckpoint
ModelCheckpoint("best_model_weights.h5", save_best_only=True)
```

---

## Requirements

```
tensorflow
numpy
scikit-learn
```


```bash
pip install -r requirements.txt
```


---

## How to Run

1. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d rijulshr/pneumoniamnist
   unzip pneumoniamnist.zip -d data/
   ```

2. Run the notebook:
   ```bash
   jupyter notebook InceptionV3_PneumoniaMNIST.ipynb
   ```

3. Evaluate and export results.

---

## Project Structure

```
.
├── data/
│   └── pneumoniamnist.npz
├── InceptionV3_PneumoniaMNIST.ipynb
├── README.md
├── inception_pneumonia_model.h5
├── best_model_weights.h5
├── requirements.txt
```

---

##  Notes

- You may fine-tune the base layers of InceptionV3 after initial training to improve generalization.
- Consider using class weights or threshold tuning to improve recall on the Normal class.