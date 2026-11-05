# Lab 07: CNN Architecture, Step by Step

Implements the forward pass of a Convolutional Neural Network from scratch in NumPy, using a small 5×5 grayscale image so every step can be checked by hand against the lecture slides.

## Pipeline
| Step | Operation | Output |
|---|---|---|
| 1 | Input image | 5×5 grid of pixel values |
| 2 | Convolution (3×3 vertical-edge kernel, stride 1, no padding) | 3×3 feature map |
| 3 | ReLU: `f(x) = max(0, x)` | 3×3, negatives set to 0 |
| 4 | Max pooling (2×2 window, stride 1) | 2×2 |
| 5 | Flatten + fully connected layer | One score per class |
| 6 | Softmax | Probabilities summing to 100% |

## Results
| Class | Score | Probability |
|---|---|---|
| Cat | 3.3 | 81% |
| Dog | 1.5 | 13% |
| Bird | 0.6 | 5% |

The outputs of each step match the lecture slides.

## Notes
- The weights and class scores are illustrative values, not trained, so the Cat/Dog/Bird result demonstrates the maths and is not a real prediction.
- A trained image classifier would use a framework such as TensorFlow/Keras or PyTorch.

## Files
- `Lab07_CNN_Architecture.ipynb`: the notebook with all outputs and the stage-by-stage plot

## Run
Open the notebook in Google Colab and use **Runtime → Run all**.

## Stack
Python, NumPy, Matplotlib
