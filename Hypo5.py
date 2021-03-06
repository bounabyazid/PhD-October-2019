#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Oct  7 11:00:00 2019
@author: Yazid BOUNAB
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mstats
from itertools import repeat
import researchpy as rp
import statsmodels.api as sm
from statistics import mean,median,stdev

from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

import matplotlib.pyplot as plt

from LoadTextData import Load_GalLery_Textual_Data, Load_GoogleVision_Labels, Load_Google_Labels
from ImgurComments import Countries,galeries

from senti_client import sentistrength

# Luxuriousness attracts user's interests

#https://pythonfordatascience.org/anova-python/

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

senti = sentistrength('EN')

Luxury = [
          ['bicsU','ePVGARS','TzZocYt','V3dW4Dn','ZqCU9wv'],
          ['6aCY1be','0fGGq','ZTGHc'],
          ['p8iAlVe','PcO48fK','SJ8Lo','xsBKJJ5','Y6qMZTX'],
          ['i56JLpa','PLn0Owa','spfNPNO','y5CDWGk'],
          ['LX9OS','sN6Y5WH'],
          ['CNVj2p5','DNF4IWn','fgb1NW8','joBhaAt','mMxSAT6'],
          ['lY15s','mwLdDmt','PpN8ZYn','PpN8ZYn'],
          ['6S9f8o7','COAND','S44YI5R','xsX5pyC'],
          ['1S0iQ1v','aiBgw4H','HRoAKgZ','JWWfhpp','kmAu4Hv','Lvh407h','NRAjuGF'],
          ['Q5I1F','YsR0y','RdQVSBB'],
          
          ['C0w9u4L','lGpGyyG'],
          ['5cIntm3','5lJ5myr','E4ALLZr','wx4VIGt'],
          ['1yWxOji','5xJiRFp','9hDeL37','hyO7iLB','Oj2Rb'],
          ['7Q87LqL','Gi0jc','v15Ie'],
          ['10HQj2K','fqqdVPR'],
          ['0ox974Z','D64oGRw','EySdGX7','f8NBx6l'],
          ['hHjflRD','ldOMnZh','mBpgfPo','QqntK','xmcvF'],
          ['3DFWMYu','CfG7H0F','hdzELGJ','n8FMpoV','QaUaE6t','qxChgJm','WnQyqfo','Zdky8JL'],
          ['rkjlGvw','sTUpBCS'],
          ['gonkB'],
          
          ['7IJXO','76Ooc','du4L6sy','ma1PO4W','R4Ty7'],
          ['6v8HmHE','CLLDz','MHrnQCe','RdgMIOZ'],
          ['1MZ2Z7l','Ajf9g3D','FDZ3lTW','HSqPaLA','iHnbahA','lceHsT6','XInPbu9'],
          ['i0mKUbv','IgIHvF5','lYUup','UtQgTH7','X6E86'],
          ['GFaDI','xYGcnRd','z9YPKuz'],
          ['0FFKc','rRKKhZF','syczAV4','TFYEj'],
          ['2lxW8ZK','Ep4Us','fgR0tGr','J0aS5e2'],
          ['GLS5sCL','HjrlFFC'],
          ['1WJ2oAA','2MMPfGr','5um4InX','OadXq','TXjepX0','U2MVdYz'],
          ['onTB5vb','ZvxtR1v'],
          
          ['ATb31Dj','h7bTe','IaV1fW9','m9QsAQh','Og6eUW0','wk9XARA'],
          ['tobnp'],
          ['IjWju','n8iaOQw','SsYSw4z'],
          ['2Vn789J','6xtXgHp','aKVjDox','iiLcAjM'],
          ['867Sv6G','5586I'],
          ['62fI8Nf','xenUl'],
          ['4fu8moV','bMklK','k8QL66z'],
          ['bO1oiuK','c2fSBez','E5A3H2p'],
          ['9MPH3','atkml','WixNSG5','Yuoe5UZ'],
          ['kAZVc','NjCzt'],
          
          ['30JQ5c4','OshTmjE','PvDi8','SFBUW08','vkIXqNa'],
          ['2xXjL','82yaC9e','kS02H9B','kYSWkmD','p3lodH8','QhjBRh9'],
          ['1YjteIr','gUjLLN3','VT3Hd'],
          ['Aeh8MFi','B36odyf'],
          ['e3U8O','Ice3sai','T22O5TO','vcwrYpX'],
          ['4EUZ6mi','9hOdzsx','dAKfsJx','Q89Sg','SlTRt'],
          ['24e466c','92JMtKi','EkvaUDH','Es7VHhT','gCVE8J5','o0OauLV','WG1Yg1H','YvjbdNW'],
          ['75rbKac','ahPtwMS','bvxZ8Th','djCGk6D','MTRo5','vk6QXAM']
          ]

def Senti_List(List):
    Score = []
    for label in List:
        res = senti.get_sentiment(label)
        Score.append(res['neutral'])
    return Score

def Luxury_Labels():
    Luxurykeys = []
    i = 0
       
    for Country in sorted(Countries):
        #print(str(i+1) + ' : ' + Country)
        for gallery_id in Luxury[i]:
            if os.path.isdir(DataSet+'/'+Country+'/'+gallery_id):
               #print(gallery_id) 
               Labels,jData = Load_Google_Labels(Country, gallery_id)
               Luxurykeys.extend(Labels)
        i+=1
    
    df = pd.DataFrame(data={"Luxury keys": sorted(list(set(Luxurykeys)))})
    df.to_csv("Luxury keys.csv", sep=',',index=False, encoding="utf-8")

def Luxury_vs_users():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    
    NbComments = []
    i = 0
    for Country in Countries:
        print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Labels,jData = Load_Google_Labels(Country, Galeries_Matrix[i,j])
            for label in Labels:
                if label in Luxurykeys:
                   Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
                   NbComments.append(len(Comments))
                
        i+=1
    return NbComments

def Luxury_vs_NonLuxury(Sentiment=False):
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    LuxuryList = [item for sublist in Luxury for item in sublist]

    NbComments = []
    Groups = []
    Sentiments = []
    
    i = 0
    for Country in Countries:
        #print(str(i+1) + ' : ' + Country)
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            NbComments.append(len(Comments))
            if Galeries_Matrix[i,j] in LuxuryList:
               Groups.append('Luxary') 
            else:
                Groups.append('NonLuxuary')
            if Sentiment:
               Sentiments.append(Senti_List(Comments))
        i+=1
    if Sentiment:
       return Groups,NbComments,Sentiments
    else:
        return Groups,NbComments

def Hypo5():
    Groups,NbComments = Luxury_vs_NonLuxury(False)
    
    df = pd.DataFrame({'Groups':Groups,'NbComments':NbComments})
    
    
    print(stats.f_oneway(df['NbComments'][df['Groups'] == 'Luxary'], 
             df['NbComments'][df['Groups'] == 'NonLuxuary']))
   
    #df['Groups'].replace({1: 'Luxary', 2: 'NonLuxuary'}, inplace= True)
        
    print(stats.kruskal(Groups,NbComments))
    #print(stats.kruskal(df['Groups'].tolist(),df['NbComments'].tolist()))
    maov = MANOVA.from_formula('Groups ~ C(NbComments)', data=df)
    print(maov.mv_test())

    results = ols('NbComments ~ Groups', data=df).fit()
    print(results.summary())
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    return df

def KruskalTest(Type = 'NbComments'):   
    if Type == 'NbComments':
       Groups,NbComments = Luxury_vs_NonLuxury(False)

       df = pd.DataFrame({'Groups':Groups,'NbComments':NbComments})
       df['Groups'].replace({'Luxary': 1, 'NonLuxuary': 2}, inplace= True)
      
       Col_1 = df['NbComments'].tolist()
       Col_2 = df['Groups'].tolist()    
    else:
        Groups,NbComments,Sentiments = Luxury_vs_NonLuxury(True)
        SGroups = []
        for i in range(0,len(Groups)):
            SGroups.extend(repeat(Groups[i], len(Sentiments[i])))
        GSentiments = [float(item) for sublist in Sentiments for item in sublist]
        df = pd.DataFrame({'Groups':SGroups,'Sentiments':GSentiments})
        df['Groups'].replace({'Luxary': 1, 'NonLuxuary': 2}, inplace= True)

        Col_1 = df['Sentiments'].tolist()
        Col_2 = df['Groups'].tolist()    
    print("Kruskal Wallis H-test "+Type+" test:")
 
    H, pval = mstats.kruskalwallis(Col_1, Col_2)

    print("H-statistic:", H)
    print("P-Value:", pval)

    if pval < 0.05:
       print("Reject NULL hypothesis - Significant differences exist between groups.")
    if pval > 0.05:
       print("Accept NULL hypothesis - No significant difference between groups.")

    return df

def Mean_Std(Type = 'NbComments'):
    print('____________________________________________')
    if Type == 'NbComments':
       Groups,NbComments = Luxury_vs_NonLuxury(False)

       df = pd.DataFrame({'Groups':Groups,'NbComments':NbComments})
       df['Groups'].replace({'Luxary': 1, 'NonLuxuary': 2}, inplace= True)
       
       NbCommentsLuxury = list(df[df['Groups']==1]['NbComments'].values)
       print('Luxury : mean = ',mean(NbCommentsLuxury),' meadian = ',median(NbCommentsLuxury),' stdev = ',stdev(NbCommentsLuxury))
       
       NbCommentsNoLuxury = list(df[df['Groups']==2]['NbComments'].values)
       print('Non Luxury : mean = ',mean(NbCommentsNoLuxury),' meadian = ',median(NbCommentsNoLuxury),' stdev = ',stdev(NbCommentsNoLuxury))

    else:
        Groups,NbComments,Sentiments = Luxury_vs_NonLuxury(True)
        SGroups = []
        for i in range(0,len(Groups)):
            SGroups.extend(repeat(Groups[i], len(Sentiments[i])))
        GSentiments = [float(item) for sublist in Sentiments for item in sublist]
        df = pd.DataFrame({'Groups':SGroups,'Sentiments':GSentiments})
        df['Groups'].replace({'Luxary': 1, 'NonLuxuary': 2}, inplace= True)

        NbSentimentsLuxury = list(df[df['Groups']==1]['Sentiments'].values)
        print('Luxury : mean = ',mean(NbSentimentsLuxury),' meadian = ',median(NbSentimentsLuxury),' stdev = ',stdev(NbSentimentsLuxury))
       
        NbSentimentsNoLuxury = list(df[df['Groups']==2]['Sentiments'].values)
        print('Non Luxury : mean = ',mean(NbSentimentsNoLuxury),' meadian = ',median(NbSentimentsNoLuxury),' stdev = ',stdev(NbSentimentsNoLuxury))

def Plot_Luxury_Proportion():
    LuxuryL = len([item for sublist in Luxury for item in sublist])
    NoLuxury = len(galeries)-LuxuryL
    
    #plt.bar(['Nature Images','No-Nature Images'], [NatureL,NoNature], color='g')
    plt.bar(['Luxury Images'], [LuxuryL], color='pink')
    plt.text('Luxury Images', LuxuryL, LuxuryL, fontsize=12)
    
    plt.bar(['No-Luxury Images'], [NoLuxury], color='orange')
    plt.text('No-Luxury Images', NoLuxury, NoLuxury, fontsize=12)
    
    plt.title('Proportion of Luxury and No-Luxury images in Tourism48 Dataset')
    
    #plt.xlabel()
    plt.ylabel('Number of images')
    plt.savefig('Proportion_of_Luxury_and_No_Luxury_images.eps', format='eps')
    plt.show()
    
def Plot_Luxury_NoLuxury_Proportion():
    N = 48
    
    LUXURY = []
    NOLUXURY = []
    
    Galeries_Matrix = [list(elem) for elem in list(zip(*[iter(galeries)]*10))]
    #LuxuryList = [item for sublist in Luxury for item in sublist]
    
    #CountryGaleries = dict(zip(Countries, Galeries_Matrix))
    CountryLuxury = dict(zip(Countries, Luxury))
    
    for Country in Countries:
        LUXURY.append(len(CountryLuxury[Country]))
        NOLUXURY.append(10-len(CountryLuxury[Country]))
        
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, LUXURY, width)
    p2 = plt.bar(ind, NOLUXURY, width, bottom=LUXURY)

    plt.xlabel('Tourism48 Countries')
    plt.ylabel('Galeries')
    plt.title('Luxuriousness proportion')
    #plt.xticks(ind, Countries)
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Luxury', 'No-Luxury'))
    plt.savefig('Proportion_Luxury_images.eps', format='eps')
    plt.show()
    
    return LUXURY,NOLUXURY
    
#df = KruskalTest('NbComments')

#Plot_Luxury_Proportion()
LUXURY,NOLUXURY = Plot_Luxury_NoLuxury_Proportion()
#df2 = KruskalTest('Sentiments')

#Mean_Std(Type = 'NbComments')
#Mean_Std(Type = 'Sentiments')


