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

虽然本例数据异常的情况发生很少，但是对于其他数据驱动的工程通常数据预处理需要对异常数据进行处理，常见的异常数据包括错误标注，缺失，等等，通常处理方法可以包含 1.删除含有异常值的记录； 2.将异常值视为缺失值，交给缺失值处理方法来处理； 3.用平均值来修正； 4.不处理

## Methodology

CNN have been introduced for multi-image-classify. In this project, it will explore the model in three steps:
1. build self-defined CNN from scratch;
2. transfer learning using pre-trained VGG16 as feature extractor + defined model;
3. using another models as feature extractor; (VGG19, Resnet-50, Inception, Xception)


## Model performance

To train the model, the Adam optimizer are employed, also categorical_crossentropy as loss and measure the accuracy of predictions on validation set.<br>

categorical_crossentropy损失函数，交叉熵是用来评估当前训练得到的概率分布与真实分布的差异情况。它刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。训练网络所使用的m个类别的标签值是经过矢量化后的m维向量，其中向量一个索引为1，其余索引为0，对应一个类别 (like one-hot encoding)。这样类别被向量化，和神经网络训练出来的m个概率值对应，概率值最大的那个输出所对应的向量，其所代表的标签即为所对应的。
标签向量化：keras中可使用to_categorical对标签值进行向量化。

准确率(accuracy)： 对于给定的测试数据集，分类器正确分类的样本数与总样本数之比. 也就是损失函数是0-1损失时测试数据集上的准确率. accuracy是正确预测的样本数占总预测样本数的比值，它不考虑预测的样本是正例还是负例。<br>
Accuracy = (预测正确的样本数)/(总样本数)=(TP+TN)/(TP+TN+FP+FN)

因为本任务属于多分类问题，所以评价标准和损失函数如是选择。

Here is the test accuracy after training:
1. training from scratch: 1.1962%；
2. transfer learning with VGG16: 43.7799%;
3. transfer learning with Xception: 86.9617%.

Also another observation is the training speed is much faster when using transfer learning.
After trained model, the inference is deployed through Flask.

对于模型调优，本例尝试了:

1. 使用不同的优化算子：Adam, RMSProp；

2. Batch size;

3. 不同模型： VGG19， Xception；

## Conclusions
1. data pipeline make the data ready for training. This case the preprocessing are readily avaiable using Keras, but for many real-problem, preprocessing using data pipeline is enssential. Collecting data, remove outliers image data, and normalize. Somethimes when the data is limited image augmentation technique would also be introduced.
2. transfer learning could help with model to converge faster, and also provided more accuracy results. deep learning needs to be in a scene with a large amount of labeled data in order to make better use of its effects. However, in many practical scenarios, we do not have enough labeled data; the universal model can solve most public problems, but it is difficult to meet the specific needs of individual models. Therefore, it is necessary to transform and adapt the general model to meet personalized needs; transfer knowledge from similar fields through transfer learning; model training for some massive data requires a lot of computing power. Generally, small and medium-sized enterprises or individuals Can't afford to burn this money, so they need to be able to use these data and models.
3. model could be benefited from advanced model (deeper structure). In theory, deeper CNN has stronger capacity of extracting advance or complex features, but also it could also be suffering from training difficulties such as gradient vanishing or overfitting. 

缺点：Xception参数量大，导致inference的速度缓慢。为使模型更快速推断，更高效的网络（Mobilenet， ShuffleNet）可能是潜在选择。


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

