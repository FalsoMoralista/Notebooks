#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os 


# In[2]:


# Returns the file list from a directory for a given extension.
def get_file_list(path,ext):
    file_list = []
    files = os.listdir(path)
    for f in files:
        if f.endswith(ext):
            file_list.append(f)
    return file_list
    


# In[3]:


DATA_DIR = '/Users/lucianoaraujo/Desktop/ImageCLEF2013PlantTask'
TRAIN = '/train/'
TEST = '/test/data/test'
LIST_OF_TRAINING_IMAGES = '/list_of_train_images.csv'


# In[4]:


training_images_csv = pd.read_csv(DATA_DIR+LIST_OF_TRAINING_IMAGES,delimiter=';')


# In[5]:


# Load the csv containing the training images information 
# Get the subset relating image id to species
training_images_csv = training_images_csv[['ImageID','GenusSpecies']]
# Separate the list of classes
classes_list = training_images_csv['GenusSpecies']
# Plot some data
training_images_csv


# In[ ]:


# Get dataframe of ImageID indexed by Species then
# moves all images from every class to a folder.
images_indexed_by_class = training_images_csv.set_index('GenusSpecies')
# Record the current folder index
class_folder_index = 0
for className in classes_list:
    # While iterating through classes get the list of images for each class
    class_images = images_indexed_by_class.loc[className]
    class_images = class_images.ImageID
    # Create directory
    os.mkdir(DATA_DIR + TRAIN +str(class_folder_index))
    # Iterates through images moving them into a new folder 
    for image in class_images:
        jpg = '.jpg'
        xml = '.xml'
        cat_source = DATA_DIR + TRAIN + str(image)
        cat_destiny = DATA_DIR + TRAIN + str(class_folder_index) + '/'

        img_src = cat_source + jpg
        print(img_src)
        print(cat_destiny)
        get_ipython().system(u'mv img_src cat_destiny')
        xml_src = cat_source + xml
        get_ipython().system(u'mv xml_src cat_destiny')
    class_folder_index = class_folder_index + 1    
    break


# In[ ]:





# In[ ]:




