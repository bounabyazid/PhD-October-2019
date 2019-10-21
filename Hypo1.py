#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
#https://github.com/ocampor/image-quality?source=post_page-----391a6be52c11----------------------

import pickle

import numpy as np
from statistics import mean 
import matplotlib.pyplot as plt

from ImgurComments import Countries,galeries
from LoadTextData import Load_GalLery_Textual_Data

import imquality.brisque as brisque
import PIL.Image

import os, os.path

from scipy.stats.stats import pearsonr
from senti_client import sentistrength

#  Media posters are impacted by the layout of the visual post.

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

senti = sentistrength('EN')

def Senti_List(List):
    Senti_Labels = []
    Score = []
    for label in List:
        Dict = {}
        res = senti.get_sentiment(label)
        Dict[label] = res
        Senti_Labels.append(Dict)
        Score.append(res['neutral'])
    #return Senti_Labels,Score
    return Score

def Sentiments_Analysis():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    
    Sentiments = []
    NbComments = []
    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            Sentiments.append(round(mean([float(i) for i in Senti_List(Comments)]),2))
            NbComments.append(len(Comments))
        i+=1
    return Sentiments,NbComments

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
    #NbComments = Number_of_Comments()
    Sentiments,NbComments = Sentiments_Analysis()
    
    r,p = pearsonr(NbComments, QScores)
    print('Correlation between Images quialty vs Comments Number = ',round(r,2), round(p,2))
    
    r,p = pearsonr(Sentiments, QScores)
    print('Correlation between Images quialty vs Sentiments = ',round(r,2), round(p,2))

    return QScores, NbComments, Sentiments

QScores, NbComments, Sentiments = Hypo1()
