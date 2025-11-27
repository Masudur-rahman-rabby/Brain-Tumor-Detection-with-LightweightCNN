

# Brain Tumor Detection Using Lightweight CNN

This project focuses on detecting **four types of brain tumors** from MRI images using a **Lightweight Convolutional Neural Network (CNN)** based on **MobileNetV2**.
The goal is to create a model that is **accurate, fast, and deployable even on low-resource devices** such as mobile phones or web applications.



## Project Summary

Brain tumors are life-threatening illnesses, and early detection is crucial.
Traditional diagnosis requires manual inspection of MRI scans by radiologists, which is time-consuming and prone to error.

This project uses **Deep Learning** to automatically classify MRI images into:

* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor**

The final model is:

* Lightweight
* Efficient
* Easy to train
* Easy to deploy



## Dataset Overview

The dataset used is from **Kaggle**, already organized into training and testing folders:

```
Training/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/

Testing/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/
```

Each folder contains MRI images belonging to that category.

### Why this dataset is good:

* Well-labelled
* Balanced across classes
* Image quality is consistent
* Easy to integrate into Keras/TensorFlow pipelines



## Methodology

### **1. Data Preprocessing**

* Images resized to 224×224 pixels
* Pixel values normalized
* Data augmentation used to prevent overfitting
* Separate training and testing sets

**Reasoning**:
Standardizing inputs reduces variance and improves overall accuracy while preventing the model from memorizing patterns.



### **2. Model Architecture**

A **Lightweight CNN (MobileNetV2)** is used because:

* It has very few parameters
* Optimized for real-time inference
* High accuracy while being computationally cheap
* Suitable for deployment on mobile or web

The model uses:

* Pretrained ImageNet weights
* Global average pooling
* Dropout for regularization
* Softmax output for 4-class classification



### **3. Training Strategy**

The model is trained with:

* Cross-entropy loss
* Adam optimizer
* Early stopping
* Learning rate scheduling

**Reasoning**:
This combination provides fast convergence and avoids overfitting while improving stability during training.



### **4. Evaluation**

The model is evaluated using:

* Accuracy
* Loss curves
* Confusion matrix
* Classification report

These metrics reveal how well the model distinguishes between tumor types and how reliably it performs on unseen images.



### **5. Explainability (Grad-CAM)**

Grad-CAM heatmaps are applied to visualize:

* Where the model is focusing
* Whether the model is attending to tumor regions

This provides interpretability and helps validate predictions from a medical perspective.



### **6. Deployment with Gradio**

A simple interactive interface is built using **Gradio**, allowing users to:

* Upload an MRI image
* View the predicted tumor type
* See confidence scores

This is useful for demonstrations, prototypes, and real-world testing.



## Results

* High accuracy on test data
* Excellent generalization due to augmentation
* Low model size (Lightweight)
* Real-time classification capability
* Grad-CAM supports medical interpretability



## Applications

* Early tumor detection
* Medical decision support
* Student research projects
* Mobile health apps
* Clinical workflow automation



## Future Improvements

* Use segmentation before classification
* Train on larger multi-hospital MRI datasets
* Convert model to TensorFlow Lite for mobile deployment
* Add uncertainty estimation for safer medical use



