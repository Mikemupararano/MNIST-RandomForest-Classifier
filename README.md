# MNIST-RandomForest-Classifier

This project involves building and evaluating a Random Forest classifier on the MNIST dataset using scikit-learn. The MNIST dataset is a well-known dataset in the machine learning community, consisting of handwritten digit images. This notebook includes steps for loading the dataset, preprocessing, training the model, tuning parameters, and evaluating the performance of the model.

## Overview
The mnist_task.ipynb notebook includes the following steps:

Loading the MNIST dataset: We use the load_digits function from sklearn.datasets to load the dataset.
Splitting the dataset: The dataset is split into training and test sets to evaluate the model's performance on unseen data.
Training a Random Forest Classifier: We use the RandomForestClassifier from sklearn.ensemble to build our classification model.
Parameter Tuning: We tune the n_estimators parameter to find the optimal number of trees in the forest.
Model Evaluation: The model's performance is evaluated using a confusion matrix, accuracy, precision, recall, and F1-score.
## Setup
To run this project, you need to have Python installed along with the following libraries:

numpy
scikit-learn
matplotlib
seaborn
You can install these dependencies using pip:

sh
Copy code
pip install numpy scikit-learn matplotlib seaborn
Usage
Clone the repository or download the mnist_task.ipynb notebook.

Run the notebook: You can run the notebook in a Jupyter environment. Make sure you have Jupyter installed. If not, you can install it using:

sh
Copy code
pip install notebook
Execute the cells: Open the mnist_task.ipynb notebook in Jupyter and execute the cells sequentially to load the dataset, train the model, and evaluate its performance.

## Key Results
Confusion Matrix: Visual representation of the model's performance, showing the true versus predicted labels.
Accuracy: Proportion of correctly predicted instances over the total instances.
Precision: Measure of the accuracy of the positive predictions.
Recall: Measure of the model's ability to find all the relevant cases.
F1-Score: Harmonic mean of precision and recall, providing a single metric to evaluate the model.
The results indicate the classes the model struggles with the most, providing insights into potential areas for improvement.

## Conclusion
This project demonstrates how to build and evaluate a Random Forest classifier on the MNIST dataset using scikit-learn. The notebook serves as a practical guide for data preprocessing, model training, parameter tuning, and performance evaluation in machine learning tasks.

