#importing libraries
import cv2
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras.preprocessing import image


#TASK 3:
#loading in th emodel made in task 1 and 2 using keras:
# Corrected model loading
model = keras.models.load_model("Assessment1/.model.h5")

#model.summary()


#uploading class names to identify the images in output 
class_names = ["bike", "car", "deer", "mountain"]

#loading and preprocessing now new images
#creating a function with parameters of image_path 
#outputs the predicted class name for the new image
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    #predicting images into loaded model
    predictions = model.predict(img_array)

    #finding the predicted class name:
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    #print(predictions)
    #print(img_array)

    return predicted_class



#saving new image and outputting results 
#uploading random images to predict/test my code
new_image1 = "Assessment1/table_mountain.jpeg"
new_image2 = "Assessment1/deer_image.jpeg"
new_image3 = "Assessment1/car_image.jpeg"
img4 = "Assessment1/bike_image.jpeg"
predicted_class1 = classify_image(new_image1)
predicted_class2 = classify_image(new_image2)
predicted_class3 = classify_image(new_image3)
predicted_class4 = classify_image(img4)


print(f"The predicted class for the first images is: {predicted_class1}")
print(f"The predicted class for the second images is: {predicted_class2}")
print(f"The predicted class for the third images is: {predicted_class3}")
print(f"The predicted class for the third images is: {predicted_class4}")



#1/1 [==============================] - 0s 91ms/step
#1/1 [==============================] - 0s 9ms/step
#1/1 [==============================] - 0s 10ms/step
#The predicted class for the first images is: mountain
#The predicted class for the second images is: bike
#The predicted class for the third images is: bike