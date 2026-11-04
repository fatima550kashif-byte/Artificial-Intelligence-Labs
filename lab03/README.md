# Lab 03: Transforming Text and Image Data into a Machine-Understandable Format

Course lab for **Introduction to Artificial Intelligence (AI-101L)**, done in Google Colab.

Raw text and raw images can't be fed directly to a traditional machine learning algorithm. This lab turns both into **fixed-length numeric vectors** and looks at what information each step throws away.

## What's inside

### Part A: Text features
- Bag-of-Words with `CountVectorizer`
- Document-term matrix shown as a pandas DataFrame
- Default behaviours checked: lowercasing and dropping single-character tokens
- Sparsity of the matrix
- Exercises: unigrams vs bigrams, and why raw counts favour long documents

### Part B: Image features
- Loading an image with Pillow and converting it to a NumPy array
- Grayscale conversion (3 channels reduced to 1)
- Flattening the pixel matrix into a 1D feature vector
- Resizing at 100%, 75%, 50%, 25%, 10% and 5%, to show the loss of detail
- Reusable function `image_to_features(path, size)` that returns the same vector length for any image
- Home assignment: building a labelled two-class dataset (`X`, `y`) and saving it as `dataset.npz`

## Pipeline

```
Text :  Documents -> Tokenisation -> Vocabulary -> Counting  -> Document-Term Matrix
Image:  RGB Image -> Resize       -> Grayscale  -> Flatten   -> Feature Vector
```

## Tech stack

Python, scikit-learn, Pillow, NumPy, pandas, Matplotlib, Google Colab

## How to run

1. Open `Lab_03_Text_and_Image_Features.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload your own image (for example `me.jpeg`) when the upload cell asks, or put it in Google Drive and set the path.
3. For the home assignment, add your own pictures to `data/class_a/` and `data/class_b/`.
4. Run the cells from top to bottom.

## Key takeaways

- Every transformation discards information: Bag-of-Words loses word order, and grayscale loses colour.
- Resizing all images to one size guarantees a fixed-length feature vector.
- The more an image is reduced, the fewer features it has and the less detail it keeps.
- 20 samples in 4096 dimensions is a very small dataset for the number of features.

## Author

Fatima, Robotics and AI, University of Lahore
