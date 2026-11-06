# Lab 08: Handwritten Digit Recognition with a CNN

A Convolutional Neural Network built with TensorFlow/Keras that classifies handwritten digits (0 to 9) from the Kaggle Digit Recognizer dataset (MNIST format, 42,000 labelled 28×28 images).

## Pipeline
1. Load `train.csv` (784 pixel columns plus a label)
2. Normalise pixels to 0 to 1 and reshape to 28×28×1
3. Stratified 80/20 train/validation split
4. Build, compile and train the CNN (Adam, sparse categorical cross-entropy, 10 epochs, batch size 64)
5. Evaluate with accuracy/loss curves, a confusion matrix and a classification report
6. Save the model as `mnist_cnn_model.keras` and reload it
7. Predict on a custom digit image, preprocessed into MNIST format

## Model architecture
| Layer | Output shape |
|---|---|
| Conv2D (32 filters, 3×3) + ReLU | 26×26×32 |
| MaxPooling 2×2 | 13×13×32 |
| Conv2D (64 filters, 3×3) + ReLU | 11×11×64 |
| MaxPooling 2×2 | 5×5×64 |
| Flatten | 1600 |
| Dense 128 + ReLU, Dropout 0.3 | 128 |
| Dense 10 + Softmax | 10 |

Total parameters: 225,034

## Results
| Metric | Value |
|---|---|
| Training accuracy (epoch 10) | 99.5% |
| Validation accuracy (epoch 10) | 98.9% |
| Validation loss (epoch 10) | 0.045 |

## Files
- `Lab08_Digit_Recognition_CNN.ipynb`: notebook with outputs and plots
- `mnist_cnn_model.keras`: trained model

## Run
Open the notebook in Google Colab, upload `train.csv` when prompted (from https://www.kaggle.com/c/digit-recognizer/data), then use **Runtime → Run all**.

## Notes
- Custom images work best as a thick dark digit on plain white paper. Photos with shadows or messy handwriting can give wrong predictions.
- The model reads one digit at a time. Multi-digit numbers need the images split into separate digits first.

## Stack
Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, seaborn
