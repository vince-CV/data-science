# Dog Breed Detector and Classify Project

This project is capstone project for DSND. It will using the Dog species images to train a Doggie classifer, make an interestig application along with human face detection, and finally deploy the model inference online.

Dog species identification is a hard tasl even for human. So the major goal to this project are:
1. explore the data, including data/outliers preprocess;
2. build the CNN model to tackle this probelm;
3. using training strategies for a higher model performance;
4. deploy a ML web application online for inference. 

In order to train and evaluate the performance of the CNN model, this project will use the Adam optimizer, categorical_crossentropy as loss function, and accuracy as the metrices.
It is reasonable to categorical cross entropy to measure the loss because our objects could be more than 100 classes and it's a classification problem. So accuracy is straitforward to evaluathe the classify tasks.

## Project Analysis

The data set is open-sourced and it will using keras.preprocessing to handle the preprocessing part. Bacause one of the tasks of this project is to detect human faces in the images so we should be aware of human faces in the dog set.
Before training model we should have a feel how the face detetor performed on dog data set: (opencv facial detector, haarcascades)
1. 99.0% human faces successfully deteted using on human faces images when using human face detector;
2. 12.0% human faces mistakenly detected using on dog images set when using human face detector
and how the dog detector worked on both data set (Resnet50, pretrained on ImageNet):
1. 1.0% dog misclassified calssified on human faces data set;
2. 100.0% dog identified on dag image set.

## Methodology

CNN have been introduced for multi-image-classify. In this project, it will explore the model in three steps:
1. build self-defined CNN from scratch;
2. transfer learning using pre-trained VGG16 as featre extractor + defined model;
3. using another models as feature extractor; (VGG19, Resnet-50, Inception, Xception)

## Model performance

To train the model, the Adam optimizer are employed, also categorical_crossentropy as loss and measure the accuracy of predictions on validation set.
Here is the test accuracy after training:
1. training from scratch: 1.1962%；
2. transfer learning with VGG16: 43.7799%;
3. transfer learning with Xception: 86.9617%.

Also another observation is the training speed is much faster when using transfer learning.
After trained model, the inference is deployed through Flask.

## Conclusions
1. data pipeline make the data ready for training. This case the preprocessing are readily avaiable using Keras, but for many real-problem, preprocessing using data pipeline is enssential. Collecting data, remove outliers image data, and normalize. Somethimes when the data is limited image augmentation technique would also be introduced.
2. transfer learning could help with model to converge faster, and also provided more accuracy results. deep learning needs to be in a scene with a large amount of labeled data in order to make better use of its effects. However, in many practical scenarios, we do not have enough labeled data; the universal model can solve most public problems, but it is difficult to meet the specific needs of individual models. Therefore, it is necessary to transform and adapt the general model to meet personalized needs; transfer knowledge from similar fields through transfer learning; model training for some massive data requires a lot of computing power. Generally, small and medium-sized enterprises or individuals Can't afford to burn this money, so they need to be able to use these data and models.
3. model could be benefited from advanced model (deeper structure). In theory, deeper CNN has stronger capacity of extracting advance or complex features, but also it could also be suffering from training difficulties such as gradient vanishing or overfitting. 


## What's in it
- Here are all the files in this project:
<pre>
app                                 -> the flask app dir
bottleneck_features                 -> the bottleneck features dir
dog_breed.html                      -> the html export from notebook
dog_breed.ipynb                     -> the main notebook
dog_images                          -> the dataset dir
extract_bottleneck_features.py      -> some useful functions
haarcascades                        -> haarcascades dir
images                              -> test images
README.md                           -> readme file
</pre>

## How to run it
- My Python version:
<pre>
Python 3.6.6
</pre>

- This is my pypi list:
<pre>
Package                 Version
----------------------- ---------
Flask                   1.1.2
h5py                    2.8.0
Keras                   2.3.1
matplotlib              3.1.3
mistune                 0.8.4
nltk                    3.4.5
numpy                   1.18.1
opencv-python           4.2.0.34
Pillow                  7.1.2
pip                     20.0.2
scikit-learn            0.22
scipy                   1.4.1
tensorflow-gpu          1.13.1
tqdm                    4.46.0
</pre>

- To run web application after installed all the requirements pkgs, lets run the deployed Flask app by running:
<pre>
1. cmd line: cd app
2. cmd line: python run.py
3. Running on http://127.0.0.1:3001/
</pre>

- So you are welcomed to upload your image, and have fun!

Reference and Thanks:
1. https://github.com/kylechenoO/Dog_Breed;
2. https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
2. https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

