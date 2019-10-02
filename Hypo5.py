#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
import numpy as np
from statistics import mean 
import matplotlib.pyplot as plt
from LoadTextData import Load_GalLery_Textual_Data

from ImgurComments import Countries,galeries

from senti_client import sentistrength
from scipy.stats.stats import pearsonr

# Bening Envy

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

def Sentiments_Analysis(Threshold):
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    
    Sentiments = []
    NbComments = []
    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            S = round(mean([float(i) for i in Senti_List(Comments)]),2)
            if abs(S) >= Threshold:
               Sentiments.append(S)
               NbComments.append(len(Comments))
        i+=1
    return Sentiments,NbComments

def Hypo5():
    Sentiments,NbComments = Sentiments_Analysis(3)
    r,p = pearsonr(NbComments, Sentiments)
    return round(r,2)

#r = Hypo5()
Sentiments,NbComments = Sentiments_Analysis(0.5)