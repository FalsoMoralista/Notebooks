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
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import csv

print(tf.__version__) # compatible with tensorflow 1.14

resnet50 = ResNet50(weights='imagenet')
model = Model(input=resnet50.input, output=resnet50.get_layer('avg_pool').output)
resnet50.trainable = False

base_dir = '/home/luciano/Desktop/PlantCLEF2013/'
sample = base_dir + '36310'

img = image.load_img(sample+'.jpg', target_size=(224,224))

img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

doc = minidom.parse(sample+'.xml')

plant_class = doc.getElementsByTagName('ClassId')
filename = 'resnet50_features'+'.csv'

fields = []

for i in range(2048):
    fields.append('n'+str(i))

fields.append('category')
fields.append('image name')

features = model.predict(img_data)

line = []

feature_list = list(features)

for i in len(feature_list):
    print(feature_list[i])

line.append(plant_class[0].firstChild.data)
line.append('3610')

print(line)

with open(filename,'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerow(line)    

#import pandas as pd
#featuress = pd.read_csv('resnet50_features.csv')

#for filename in os.listdir('/home/luciano/Desktop/PlantCLEF2013'):
#    if filename.endswith(".jpg"):
#    	name = filename.replace('.jpg','')
#    	doc = minidom.parse('/home/luciano/Desktop/PlantCLEF2013/'+name+'.xml')
#    	plant_class = doc.getElementsByTagName('ClassId')
#    	print(plant_class[0].firstChild.data)
#        img = image.load_img('/home/luciano/Desktop/PlantCLEF2013/'+filename, target_size=(224,224))
#        img_data = image.img_to_array(img)
#        img_data = np.expand_dims(img_data, axis=0)
#        img_data = preprocess_input(img_data)
#        features = model.predict(img_data)
#        print(features.shape)
#        print(features)

#import matplotlib.pyplot as plt

#url = 'https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg'
#name = url.split("/")[-1]
#image_path = tf.keras.utils.get_file(name, origin=url)

#raw_img = download(url)
#img = tf.image.resize(raw_img, (224,224))








