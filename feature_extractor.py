# -*- coding: utf-8 -*-
"""feature_extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ywWQyo0O8n8XDIN9-tJGKEIjPva6XObe
"""

#!pip install -q tensorflow==2.0.0-alpha0
#!pip install -q tensorflow==1.14


import os
from xml.dom import minidom
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import csv
import sys




argv = sys.argv[1:]

if len(argv) < 2:
    print('Usage: python feature_extractor.py  arg0 arg1 ')
    print('arg0: model')
    print('arg1: dataset directory ')
    print('eg: python feature_extractor.py resnet50 /home/joe/Desktop/dataset')
    print("Currently available implementations:")
    print(".Resnet50")
else:
    print(tf.__version__) # compatible with tensorflow 1.14
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    base_dir = argv[1]
    if argv[0] == 'resnet50':
        # Load pre-trained model
        resnet50 = ResNet50(weights='imagenet')
        model = Model(input=resnet50.input, output=resnet50.get_layer('avg_pool').output)
        resnet50.trainable = False
        # Format the csv fields
        fields = []
        for i in range(2048):
            fields.append('n'+str(i))
        fields.append('category')
        fields.append('image name')
        # Open the csv and append primary fields
        with open('resnet50_features.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for filename in os.listdir(base_dir): # For each image
                if filename.endswith(".jpg"):
                    print(filename)
                    name = filename.replace('.jpg','') # Get its name
                    doc = minidom.parse(base_dir+'/'+name+'.xml') # Open its xml
                    plant_class = doc.getElementsByTagName('ClassId') # Get its class
                    img = image.load_img(base_dir+'/'+filename, target_size=(224,224)) # Load and format to the model expected input size
                    img_data = image.img_to_array(img) 
                    img_data = np.expand_dims(img_data, axis=0) # Fit it to a 1D list
                    img_data = preprocess_input(img_data) # Preprocess accordingly to the model preferences
                    features = model.predict(img_data) # Get the features
            line = []
            for x in np.nditer(features): 
                line.append(str(x)) # Append them to a list format
            line.append(plant_class[0].firstChild.data) # Add its class
            line.append(name) # Add the file name
            csvwriter.writerow(line) # Finally write it down to the csv    
    if argv[0] == 'resnet152v2':
        resnet152v2 = ResNet152V2(weights='imagenet')
        model = Model(input=resnet152v2.input, output=resnet152v2.get_layer('avg_pool').output)
        resnet152v2.trainable = False
        fields = []
        for i in range(2048):
            fields.append('n'+str(i))
        fields.append('category')
        fields.append('image name')
        # Open the csv and append primary fields
        with open('resnet152v2.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for filename in os.listdir(base_dir): # For each image
                if filename.endswith(".jpg"):
                    print(filename)
                    name = filename.replace('.jpg','') # Get its name
                    doc = minidom.parse(base_dir+'/'+name+'.xml') # Open its xml
                    plant_class = doc.getElementsByTagName('ClassId') # Get its class
                    img = image.load_img(base_dir+'/'+filename, target_size=(224,224)) # Load and format to the model expected input size
                    img_data = image.img_to_array(img) 
                    img_data = np.expand_dims(img_data, axis=0) # Fit it to a 1D list
                    img_data = preprocess_input(img_data) # Preprocess accordingly to the model preferences
                    features = model.predict(img_data) # Get the features
            line = []
            for x in np.nditer(features): 
                line.append(str(x)) # Append them to a list format
            line.append(plant_class[0].firstChild.data) # Add its class
            line.append(name) # Add the file name
            csvwriter.writerow(line) # Finally write it down to the csv    
        













