#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
import os
import json
import pandas

import numpy as np
from statistics import mean 
import matplotlib.pyplot as plt
from LoadTextData import Load_GalLery_Textual_Data, Load_GoogleVision_Labels, Load_Google_Labels

from ImgurComments import Countries,galeries

from senti_client import sentistrength
from scipy.stats.stats import pearsonr

# Users appreciate more nature like pics.

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

Nature = [['7yDE0','9A5rQXR','bicsU','ePVGARS','TzZocYt','ZqCU9wv'],
          ['0fGGq','AjChT','ZTGHc'],
          ['a1C8u','p8iAlVe','PcO48fK','SJ8Lo','vmIi3','Y6qMZTX','zkahmbF'],
          ['i56JLpa','MrelgvY','PLn0Owa','spfNPNO','X2Cn4ZU'],
          ['LX9OS','pBN3g','sN6Y5WH'],
          ['CNVj2p5','DNF4IWn','mMxSAT6'],
          ['E2cJkFX','fJCxZTa','lY15s','mK6dWzg','mwLdDmt'],
          ['6S9f8o7','COAND','V4G3HQx','xsX5pyC','xyfJufu'],
          ['aiBgw4H','HRoAKgZ','JWWfhpp','kmAu4Hv','Lvh407h'],
          ['HGnho','Q5I1F','YsR0y'],
          
          ['C0w9u4L','lFScS0P'],
          ['5cIntm3','5lJ5myr','E4ALLZr','G0jK5iu','wx4VIGt'],
          ['5xJiRFp','9hDeL37','cwLSAoz','DW0iIDI'],
          ['6WY1zk5','7Q87LqL','50oq7R3','v15Ie'],
          ['0XQ3MY9','3MJxF5e','fqqdVPR','NOM2vxP'],
          ['0ox974Z','D64oGRw','EySdGX7','f8NBx6l','GJH0o','RteihhY'],
          ['ldOMnZh','mBpgfPo'],
          ['hmpah'],
          ['2VuKN','rkjlGvw','sTUpBCS','V7XPZ'],
          ['phuY3i5'],
          
          ['7IJXO','8N7vk','d7XYT6X','du4L6sy','J5hHYue','LZnJJ9s','QxcPfDw','R4Ty7'],
          ['CLLDz','j3G6T','MHrnQCe','RdgMIOZ'],
          ['1MZ2Z7l','Ajf9g3D','BAVZRRs','HSqPaLA','lceHsT6','zrPRVuN'],
          ['i0mKUbv','IgIHvF5','lYUup','rA59e49','tfs1Gp9','UtQgTH7'],
          ['hwdyO','YPwRZWv'],
          ['0FFKc','Kk9n2hL','rRKKhZF','TFYEj','TvMWE','VyRCtdr'],
          ['2lxW8ZK','GKwiiWK','J0aS5e2'],
          ['5jCE0','GLS5sCL','ULGpV'],
          ['1WJ2oAA','5um4InX','bm5d9tr','OadXq','TXjepX0'],
          ['9cY508M','onTB5vb','wlPK7','ZvxtR1v'],
          
          ['h7bTe','IaV1fW9','lACN4if','mETwduJ','Og6eUW0'],
          ['3dlQB8u','cK47uRK','o6BjCHi','XSC7wVS'],
          ['n8iaOQw','SsYSw4z','xnWps3C'],
          ['aKVjDox','CxvV6A5','iiLcAjM','yyEb3X2'],
          ['zZ2o7'],
          ['a8qifsj','xenUl'],
          ['k8QL66z','xM2YZ4r'],
          ['E5A3H2p','mAJR61Q'],
          ['9MPH3','hftKF','jwiI9mp','WixNSG5','Yuoe5UZ'],
          ['IvUl80S','JyA7N'],
          
          ['c2GTjSv','OshTmjE','PvDi8'],
          ['2xXjL','fYYulaI','H7nvQFo','HyJkbMv','kYSWkmD'],
          ['1YjteIr','Ts5W6to','TzomF','VT3Hd','ZTgIsMx'],
          ['08qa6','m1Wg9Vt'],
          ['o4PTnPV','T22O5TO','TvHgfey','VdyH6'],
          ['0GqOUs6','9hOdzsx','dAKfsJx','eRM6hHb','LrEK3o8','SlTRt','vRLD62o'],
          ['24e466c','EkvaUDH','Es7VHhT','o0OauLV','YvjbdNW'],
          ['75rbKac','MTRo5','skJPplY']
         ]

def Nature_Labels():
    Naturekeys = []
    i = 0
       
    for Country in sorted(Countries):
        #print(str(i+1) + ' : ' + Country)
        for gallery_id in Nature[i]:
            if os.path.isdir(DataSet+'/'+Country+'/'+gallery_id):
               #print(gallery_id) 
               Labels,jData = Load_Google_Labels(Country, gallery_id)
               Naturekeys.extend(Labels)
        i+=1
    
    df = pandas.DataFrame(data={"Nature keys": sorted(list(set(Naturekeys)))})
    df.to_csv("Nature keys.csv", sep=',',index=False, encoding="utf-8")

def NaturePics_Vs_Comments():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    NatureList = [item for sublist in Nature for item in sublist]
    
    NaturePics = {}
    NoNaturePics = {}
    
    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            if Galeries_Matrix[i,j] in NatureList:
               NaturePics[Galeries_Matrix[i,j]] = len(Comments)
            else:
                NoNaturePics[Galeries_Matrix[i,j]] = len(Comments)
        i+=1
    return NaturePics ,NoNaturePics 

def Hypo2():
    NaturePics ,NoNaturePics = NaturePics_Vs_Comments()
    print('Nature = ',round(mean(NaturePics.values()),2))
    print('Non Nature = ',round(mean(NoNaturePics.values()),2))
    return NaturePics ,NoNaturePics
NaturePics ,NoNaturePics = Hypo2()
