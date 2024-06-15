Fashion MNIST Classification with Convolutional Neural Network

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images of 10 different types of clothing items, each sized at 28x28 pixels.

Here's link to download dataset:
https://www.kaggle.com/datasets/zalando-research/fashionmnist

Project Features

Data Loading and Preprocessing:
- Load dataset from CSV files.
- Normalize pixel values to [0, 1].
- Visualize sample images and their labels.
- Split data into training, validation, and test sets.

Model Architecture:
- Input layer: 28x28x1.
- Convolutional layers with ReLU activation.
- MaxPooling layers for down-sampling.
- Dense layer with ReLU activation and Dropout for regularization.
- Output layer with softmax activation for multi-class classification.

Model Training:

- Compile model with Adam optimizer and sparse categorical cross-entropy loss.
- Train model with early stopping and model checkpointing.

Evaluation:

- Test accuracy on unseen data.
- Confusion matrix to visualize classification performance.
- Classification report with precision, recall, and F1-score.

Visualization:

- Plot training and validation accuracy and loss.

Dependencies:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Results:
Achieved a test accuracy of 91.34% with detailed metrics and visualizations provided in the repository.

Acknowledgements:
- Fashion MNIST dataset provided by Zalando Research.
- TensorFlow and Keras documentation.

Contact:
For any questions or suggestions, please contact stachowiak.aleksandra.18@gmail.com

