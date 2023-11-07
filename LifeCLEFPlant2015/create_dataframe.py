#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:55:04 2021

@author: lucianoaraujo
"""
import os
import pandas as pd

import xml.etree.ElementTree as ET
data = []
for xml in os.listdir('train'):
  if xml.endswith('.xml'):
    tree = ET.parse('train/'+xml)
    root = tree.getroot()
    filename = root[1].text
    content = root[3].text
    family = root[5].text
    genus = root[6].text
    species = root[7].text
    data.append([filename,content,family,genus,species])
df = pd.DataFrame(data=data,columns=['Filename','Content','Family','Genus','Species'])

df.to_csv("training_dataset_info.csv", encoding='utf-8')
