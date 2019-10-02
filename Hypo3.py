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

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

senti = sentistrength('EN')

luxury = ['bicsU','ePVGARS','TzZocYt','V3dW4Dn','ZqCU9wv',
          '0fGGq','ZTGHc',
          'p8iAlVe','PcO48fK','SJ8Lo','xsBKJJ5','Y6qMZTX',
          'i56JLpa','PLn0Owa','spfNPNO','y5CDWGk',
          'LX9OS','sN6Y5WH',
          'CNVj2p5','DNF4IWn','fgb1NW8','joBhaAt','mMxSAT6',
          'lY15s','mwLdDmt','PpN8ZYn','PpN8ZYn',
          '6S9f8o7','COAND','S44YI5R','xsX5pyC',
          '1S0iQ1v','aiBgw4H','HRoAKgZ','JWWfhpp','kmAu4Hv','Lvh407h','NRAjuGF',
          'Q5I1F','YsR0y',
          'C0w9u4L','lGpGyyG',
          '5cIntm3','5lJ5myr','E4ALLZr','wx4VIGt',
          '1yWxOji','5xJiRFp','9hDeL37','hyO7iLB','Oj2Rb',
          '7Q87LqL','Gi0jc','v15Ie',
          '10HQj2K','fqqdVPR',
          '0ox974Z','D64oGRw','EySdGX7','f8NBx6l',
          'hHjflRD','ldOMnZh','mBpgfPo','QqntK','xmcvF',
          '3DFWMYu','CfG7H0F','hdzELGJ','n8FMpoV','QaUaE6t','qxChgJm','WnQyqfo','Zdky8JL',
          'rkjlGvw','sTUpBCS',
          'gonkB',
          '7IJXO','76Ooc','du4L6sy','ma1PO4W','R4Ty7'
          '6v8HmHE','CLLDz','MHrnQCe','RdgMIOZ',
          '1MZ2Z7l','Ajf9g3D','FDZ3lTW','HSqPaLA','iHnbahA','lceHsT6','XInPbu9',
          'i0mKUbv','IgIHvF5','lYUup','UtQgTH7','X6E86',
          'rGcIdc1','xYGcnRd','z9YPKuz',
          '0FFKc','rRKKhZF','syczAV4','TFYEj',
          '2lxW8ZK','Ep4Us','fgR0tGr','J0aS5e2',
          'GLS5sCL','HjrlFFC',
          '1WJ2oAA','2MMPfGr','5um4InX','OadXq','TXjepX0','U2MVdYz',
          'onTB5vb','ZvxtR1v',
          'ATb31Dj','0Y5yPc8','IaV1fW9','m9QsAQh','Og6eUW0','wk9XARA',
          'tobnp',
          'hq8vRF0','n8iaOQw','SsYSw4z',
          '2Vn789J','6xtXgHp','aKVjDox','iiLcAjM',
          '867Sv6G','5586I',
          '62fI8Nf','xenUl',
          '4fu8moV','bMklK','k8QL66z',
          'bO1oiuK','c2fSBez','E5A3H2p',
          '9MPH3','atkml','WixNSG5','Yuoe5UZ',
          'kAZVc','NjCzt',
          '30JQ5c4','OshTmjE','jVAjNwr','SFBUW08','vkIXqNa',
          '2xXjL','82yaC9e','kS02H9B','kYSWkmD','p3lodH8','QhjBRh9',
          '1YjteIr','gUjLLN3','VT3Hd',
          'Aeh8MFi','B36odyf',
          'e3U8O','Ice3sai','T22O5TO','vcwrYpX',
          '4EUZ6mi','9hOdzsx','dAKfsJx','Q89Sg','SlTRt',
          '24e466c','92JMtKi','EkvaUDH','Es7VHhT','gCVE8J5','o0OauLV','WG1Yg1H','YvjbdNW',
          '75rbKac','ahPtwMS','bvxZ8Th','djCGk6D','MTRo5','vk6QXAM'
          ]

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

def Hypo2():
    Sentiments,NbComments = Sentiments_Analysis()
    r,p = pearsonr(NbComments, Sentiments)
    return r