# Dog Breed Detector Project
- This is the project of the final project for the Data Scientist Nanodegree.
- I choose the Dog Breed Project at all.

## What's in it
- Ok, let's have a look of all the file in this project.
<pre>
app                                 -> the flask app dir
bottleneck_features                 -> the bottleneck features dir
dog_app.html                        -> the html export from notebook
dog_app.ipynb                       -> the main notebook
dog_images                          -> the dataset dir
extract_bottleneck_features.py      -> some useful functions
haarcascades                        -> haarcascades dir
images                              -> some test images
LICENSE                             -> license file
project.md                          -> the project describe
README.md                           -> you are reading this file
</pre>

## How to run it
- At 1st, this is my python info:
<pre>
nbusr@server:~/Dog_Breed$ python3 -V
Python 3.6.6
</pre>

- This is my pypi env list:
<pre>
Package                 Version
----------------------- ---------
Flask                   1.0.2
Flask-RESTful           0.3.6
h5py                    2.8.0
image                   1.5.24
Keras                   2.2.0
matplotlib              2.2.2
mistune                 0.8.3
nltk                    3.3
numpy                   1.14.5
opencv-python           3.4.1.15
Pillow                  5.1.0
pip                     18.1
scikit-learn            0.19.1
scipy                   1.1.0
tensorflow-gpu          1.8.0
tqdm                    4.23.4
</pre>

- After installed all the pkgs before, lets start the flask app:
<pre>
nbusr@server:~/Dog_Breed$ cd app/
nbusr@server:~/Dog_Breed/app$ python3 run.py
Using TensorFlow backend.
 * Serving Flask app "run" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
 * Restarting with stat
Using TensorFlow backend.
 * Debugger is active!
 * Debugger PIN: 246-660-102
</pre>

- And after this you can open the browser to access http://localhost:5000
- You can upload your image, it will automate detector it if there's some face or dog.
