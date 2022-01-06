
# Face Recognition

I did this project about 4 years back during my bachelor's . I forgot to push it on github that time so doing it now ;)

## About the Project

To make a face recognition program, first we need to train the recognizer with dataset of previously captured faces along with its ID, 
for example we have two person then first person will have ID 1 and 2nd person will have ID 2, 
so that all the images of person one in the dataset will have ID 1 and all the images of the 2nd person in the dataset will have ID 2, 
then we will use those dataset images to train the recognizer to predict the 1 of an newly presented face from the live video frame

So let’s break the program into 3 major parts:

    1.	Dataset Creator
    2.	Trainer
    3.	Detector



## Requirements

- Python

        Python is the platform on this complete project is made. The version of this platform must be 2.7 or above.
- OpenCv 

        This is the basis library used in almost every program used in this project. Its version should be either 2.x or 3.x.
- Numpy library

        Used for type conversions, NumPy is a library for the Python programming language, adding support for large, 
        multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.  
## Steps and Screenshots

#### 1. Creation of Dataset

    Let’s create the dataset generator script, open your pythonIDLE and create a new file and save it in your project folder and make sure you also 
    have the haarcascade_frontalface_default.xml file in the same folder just like in the previous post we will need to do the following first:

        i.	 cv2 library (OpenCV library)
        ii.	 create a video capture object
        iii. cascade Classifier object

Our dataset generator is going to capture few sample faces of one person from the live video frame and assign an ID to it 
and it will save those samples in a folder which we are going to create now and we will name it dataset.

#### Let’s Test It

    If we run this code now then we will see that it will capture faces from the live video and will save it in the dataset folder

![Output Screenshot](https://github.com/AyanPahari/Face-Recognization/blob/master/Screenshot/SC1.JPG)

Now we have our dataset we can now train the recognizer to learn the faces from this dataset.

#### 2. Training the Recognizer

To perform face recognition we need to train a face recognizer, using a pre labeled dataset, In my previous post we created a labeled dataset 
for our face recognition system, now it’stime to use that dataset to train a face recognizer using opencv python. First create a python “Trainer.py” file 
in the same folder where we saved out dataset generator script in the previous post, and then create a folder in the same directory name it “Trainer”, 
this is the folder where we are going to save our recognizer after training.

#### 3. Recognition

If we run the above code given the dataset of face recognizer an id with any information (in this case name is used) we want to show on the screen, 
the output screen will look like something as shown below.

![Output Screenshot](https://github.com/AyanPahari/Face-Recognization/blob/master/Screenshot/SC2.JPG)

#### 4. Linking to SQLite3 Database

In Fig. 4, an example is shown how the information is stored in database which will be printed on the output screen of our face recognizer. 
The face recognizer will show the information at the bottom left of the contour of the face as we have programmed it in the above code to be.


![Output Screenshot](https://github.com/AyanPahari/Face-Recognization/blob/master/Screenshot/SC3.JPG)
