#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from LoadTextData import Load_GalLery_Textual_Data

from ImgurComments import Countries,galeries

from senti_client import sentistrength

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

senti = sentistrength('EN')

def Sentiments():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    
    Sentiments = []

    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            path = DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'+Galeries_Matrix[i,j]+'.json'
            with open(path) as data_file:    
                 Data = json.load(data_file)
                 
        Sentiments.append(len(Data['Comments']))
        i+=1
    return Sentiments

res = senti.get_sentiment("I think it's a group of people swimming in a body of water. ")
