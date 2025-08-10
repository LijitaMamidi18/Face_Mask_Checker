
# Face\_Mask\_Checker

This is a deep learning project focused on detecting whether a person is wearing a face mask or not using image classification. It leverages a Convolutional Neural Network (CNN) model trained on a labeled dataset of masked and unmasked faces. The system is deployed using Streamlit to provide a user-friendly web interface for real-time predictions.

---

## **Objective**

To build and deploy a deep learning model that can classify input images as:

1. Mask
2. No Mask

---

## **Technologies & Tools Used**

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib (for visualization)
* Jupyter Notebook
* Streamlit (for deployment)

---

## **Workflow**

### 1. Data Collection

* Collected face images with and without masks from open-source datasets (e.g., Kaggle, RMFD).
* Applied manual filtering and balancing for better model performance.

### 2. Data Preprocessing

* Resized and normalized all images.
* Used OpenCV to detect and crop faces.
* Split dataset into training, validation, and test sets.

### 3. Model Building

* Developed a CNN using Keras with multiple convolutional and pooling layers.
* Used activation functions like ReLU and softmax for binary classification.
* Applied dropout to reduce overfitting.

### 4. Model Training

* Trained the model with appropriate batch size and learning rate.
* Used data augmentation to generalize better on unseen data.

### 5. Model Evaluation

* Evaluated the model using:

  * Accuracy
  * Precision & Recall
  * Confusion Matrix
* Achieved high accuracy and robustness on validation/test images.

### 6. Deployment

* Deployed the model using **Streamlit** for an interactive frontend.
* Users can upload an image to receive a prediction (Mask or No Mask).
* Entire app runs in a browser without the need for HTML/CSS.


