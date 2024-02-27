import numpy as np 
import pandas as pd 
import tensorflow
from tensorflow import keras 
from keras import models, layers
import glob
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt





def load_data_preprocessing():

 
    
    all_images = []
    all_labels = []


    for i,items in enumerate(glob.glob("/Users/noushinahmadvand/Documents/kapchaproject/kapcha/*/*")):
   

     img = cv2.imread(items)
     img = cv2.resize(img, (32,32))
     img = img/255.0

     all_images.append(img)

     label = items.split("/")[-2]
     all_labels.append(label)

     if i%100 ==0 :


        print("[INFO] {}/2250 processed".format(i))

    all_images = np.array(all_images)
    lb = LabelBinarizer()
    all_labels = lb.fit_transform(all_labels)

    trainX, testX, trainY , testY = train_test_split(all_images, all_labels, test_size= 0.2)

    return  trainX, testX, trainY , testY 





def mainCNN():
   
   net = models.Sequential([
                            layers.Conv2D(32, (3,3), activation= "relu", padding = "same", input_shape = (32,32,3)),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64, (3,3), activation= "relu", padding = "same"),
                            layers.MaxPooling2D((2,2)),
                            layers.Flatten(),
                            layers.Dense(32, activation= "relu"),
                            layers.Dense(9, activation= "softmax")
                

                            ])


   net.compile(loss= "categorical_crossentropy",
               optimizer="SGD",
               metrics=["accuracy"]
               )
   return net


def show_result(h):
   plt.plot(h.history["accuracy"],label = "train accuracy")
   plt.plot(h.history["val_accuracy"],label = "test accuracy")
   plt.plot(h.history["loss"],label = "train loss")
   plt.plot(h.history["val_loss"],label = "test loss")
   plt.legend()
   plt.xlabel("epochs")
   plt.ylabel("loss/accuracy")
   plt.title("digit classification")
   plt.show()


trainX, testX, trainY , testY  = load_data_preprocessing()

net = mainCNN()
  

h = net.fit(x= trainX ,y= trainY , epochs=  20 , batch_size= 32, validation_data= (testX, testY) ) 


show_result(h)
net.save("digitclassify.h5")

