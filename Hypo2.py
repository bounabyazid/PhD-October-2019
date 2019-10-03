#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
import json

import numpy as np
from statistics import mean 
import matplotlib.pyplot as plt
from LoadTextData import Load_GalLery_Textual_Data

from ImgurComments import Countries,galeries

from senti_client import sentistrength
from scipy.stats.stats import pearsonr

# Users appreciate more nature like pics.

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

def NaturePic():
    #there must be a list of keywords that represent nature pictures
    return True

def NaturePics_Vs_Comments():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    
    Sentiments = []
    NbComments = []
    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            Sentiments.append(round(mean([float(i) for i in NaturePic(Comments)]),2))
            NbComments.append(len(Comments))
        i+=1
    return Sentiments,NbComments

def Hypo2():
    Sentiments,NbComments = NaturePics_Vs_Comments()
    r,p = pearsonr(NbComments, Sentiments)
    return r

