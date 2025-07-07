# Music Genre Classfication

---

##  Project Demo
Check out the video : https://youtu.be/yPBmJIeTnw4

---
## How it works
- first we create a folderr called '*data*' and under that we create another folder called '*genres_original*'
- we then download a dataset (around 1gb) from kaggle which contains different 30 second samples of genres of music (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (around 1gb)
- we then put the different genres inside the '*genres_original*' folder
- we use a python library called *librosa* to load the audio file and then convert it into a MEL spectrogram (a way to represent sound in terms of frequency and time, 2d graph), then plot it and save it as png
- in the training model part, we first start by reducing the size of the spectrogram image to (128x128) pixels and then converts it into a numpy array and stores it, (saves the genre name as its label)
- we use X = images and Y = labels
- we then split the data (80% training, 20% testing)
- then we go to define the CNN part by applying 32 filters of size 3x3 to learn basic patterns, we also reduce image size to keep important features using maxPooling
- we do the same thing now by applying 64 filters of size 3x3 to detect deeper features, and then the same by applying 128 filters of size 3x3
- we train the model using 128 neurons
- then we use something called *Flatten* to turn the 2d matrix of X and y into a 1d vector
- we use softmax to return the probability of what genre it could potentially be
- then we train the model using *model.fit(..)*
- we save the trained model in a *.h5* file
- then we move on to the prediction part: the image is first passed into the trained model
- the model returns probabilities of what genre it could be
- the genre with the highest probability is selected
- also we save and load the LabelEncoder in a .pkl file to store the labels (like when we get a probability, we need to convert it back into a genre name)
- and then we make the frotnend in main.py
- your welcome



  ---

  Author : Param Varsha (04/06/2025)
