# Cancer Detection - CNN Model

A CNN Model

A deep learning project

## Overview:
Cancer is a highly researched area because there are so many people who suffer and/or die of cancerous disease. This study aims to improve cancer detection in lymph nodes by using computer vision machine learning technqiues. We will examine the data given in this competition to get a better understanding of it. Then, we will run multiple convolutional neural network models with the intent to be able to classify cancerous and non-cancerous cells.
This notebook covers the thought process as to how to create simple CNN models. We will create two models, one without hyperparameter tuning, and one with tuning. Finally, we will suggest ways to improve the models in future studies.

Following layouts:

* Brief Description of the Problem and Data
* Exploratory Data Analysis (EDA) - Inspect, Visualize and Clean the Data
* Describe Model Architecture
* Results and Analysis
* Conclusion

## Brief Description of the Problem and Data:

This data contains thousands of small images where the 96x96 pixel images with 3 channels, each with an identifying label and id.
We have two datasets, a training and testing set already split for us. The training set contains 220,025 unique images and the test set contains about 57,500. To use these images in a machine learning model, we are also given an identifying dataframe with two columns: 'id' which is the unique image ID correpsonding to the training directory, and 'label' which tells us the classification category. Each label is either a 0 or 1, depending whether the image is non-cancerous or cancerous.
In the competition description, we find that if at least one pixel of an image is identified as cancerous then the whole image is therefore marked with a 1, otherwise it is 0. It is important to note that we do not have any missing values in this data which will make preprocessing more efficient.

# Set paths:

1. train = '../input/histopathologic-cancer-detection/train/'
2. test = '../input/histopathologic-cancer-detection/test/'

3. data_train = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

## Exploratory Data Analysis (EDA) - Inspect, Visualize and Clean the Data:

First, we will visualize the data. Then we will clean and preprocess the data.
We can see in the histogram and pie chart below that we have 59.5% of the labels are 0 as non-cancerous images and 40.5% are labeled 1 as cancerous images. Here, we have a split which is closer to 40/60. This means that our data is unbalanced.
We also have thousands of images to train with. For this reason, we can assume we will be able to create a sufficiently performing model which identifies cancerous images.

## Describe Model Architecture:

For this model we will be using Keras library to run a convolutional neural network (CNN).
The first model we will run without tuning any hyperparameters within the model and use that as our baseline. Then, we will run a second model, tuning hyperparameters such as learning rate, batch normalization, regularization, filter size, stride, activation layers, etc.

#### First model:

* Normalize images pre-training (image/255)
* Output layer activation (sigmoid)

#### Second model contains all the first model parameters, but we also add:

* Dropout (0.1)
* Batch Normalization
* Optimization (Adam)
* Learning rate (0.0001)
* Hidden layer activations (ReLU)

## Results and Analysis:

We can see from the above plots and diagrams for each model how well they performed with the training (and validation) sets. We see that model-one seemed to steady out a bit more than our more complex model (model-two). We see in both models that the accuracy and loss does not steady. This could pertain towards the fact that we trained with very few epochs (5) and a simple CNN model with so many pictures may need more "time" to train to converge.
After submitting both trained models separately on the test set, we can see (below) how each model performed. As expected, we see that model-one (the model without hyperparameters) performed worse

## Conclusion:

Our first model was simple with no hyperparameter tuning. The second model incorporated much more tuning and a few extra layers. Both models trained for 5 epochs and performed resonably well given that they both are fairly simple. As expected, the second model did better than the first. We can see that hyperparameter tuning does indeed contribute to the model performance and can improve the model if done correctly.
Since this project is for demonstration purposes, we did not use more epochs either. Some other ideas to make a stronger model would be to transfer learning where part of the model is taken from another well-trained model, and tune hyperparameters in different ways such as strides, filter size, activation functions, learning rate, etc,


Thank you.
