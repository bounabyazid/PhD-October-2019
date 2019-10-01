#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
#https://github.com/ocampor/image-quality?source=post_page-----391a6be52c11----------------------

import pickle

import numpy as np
import matplotlib.pyplot as plt

from ImgurComments import Countries,galeries
from LoadTextData import Load_GalLery_Textual_Data

import imquality.brisque as brisque
import PIL.Image

import os, os.path

from scipy.stats.stats import pearsonr

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

def Number_of_Comments():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    NbComments = []
    i = 0
    for Country in Countries:
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            NbComments.append(len(Comments))
        i+=1
    return NbComments

def Referenssless_Image_Quality_Assessment():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)

    valid_images = [".jpg",".gif",".png"]
    
    QScores = []

    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            path = DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'
            
            for f in os.listdir(path):
                ext = os.path.splitext(f)[1]
                if ext.lower() in valid_images:    
                   img = PIL.Image.open(path+f)
                   Q = brisque.score(img)
                   QScores.append(Q)
        i+=1
    pickle_out = open("QScores.pkl","wb")
    pickle.dump(QScores, pickle_out)
    pickle_out.close()
    
    return QScores

#QScores = Referenssless_Image_Quality_Assessment()

def Hypo1():
    pickle_in = open("QScores.pkl","rb")
    QScores = pickle.load(pickle_in)
    NbComments = Number_of_Comments()
    
    return NbComments, QScores

NbComments, QScores = Hypo1()

r,p = pearsonr(NbComments, QScores)