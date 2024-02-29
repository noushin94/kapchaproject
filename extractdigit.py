
import cv2
import tensorflow
from tensorflow import keras
from keras.models import load_model
import numpy as np


# we want to extract each number then give it to cnn models that we have
    

net = load_model("/Users/noushinahmadvand/Documents/kapchaproject/digitclassify.h5")
#reading an image, convert it to gray
img = cv2.imread("/Users/noushinahmadvand/Documents/kapchaproject/00000.jpg")
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)

#binary with threshold
T , thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

cnts, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
# I used retr_external becouse with three iyt detected 7 contours
print(len(cnts))
for i in range(len(cnts)):

    x , y , w, h = cv2.boundingRect(cnts[i])

    cv2.rectangle(img, (x-5,y-5), (x+w+5, y+h+5), (0,255,0), 2)

    roi = img[y:y+h+5, x:x+w+5]
    roi = cv2.resize(roi, (32,32))
    roi = roi/255.0
    roi = np.array([roi])

    output = net.predict(roi)[0]
    max_index = np.argmax(output) +1


    print( max_index)

    
    #cv2.imshow("image", roi)
    #cv2.waitKey(0) 




