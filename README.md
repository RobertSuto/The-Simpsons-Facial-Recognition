# Computer Vision Project 2

This is a project for the Computer Vision course at the Faculty of Mathematics and Computer Science, University of Bucharest. The project consists of two tasks:

1. Face detection: given a set of training images with faces and non-faces, train a classifier to detect faces in test images.
2. Cartoon character classification: given a set of training images with four cartoon characters, train a classifier to recognize the characters in test images.

## Usage

To run the project, follow these steps:

   This will generate positive and negative examples for cartoon character classification, train a classifier, and evaluate its performance on test images. The results will be saved in the "evaluare/fisiere-solutie/Suto_Robert_311/task2" directory.

   The method used for cartoon character classification is also based on the code provided in the course labs and assignments. The approach involves generating positive and negative examples from the training images, extracting features using the Histogram of Oriented Gradients (HOG) descriptor, using a sliding window and training a classifier using the Support Vector Machine (SVM)
