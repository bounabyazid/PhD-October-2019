import re
import pickle
import emoji
import regex
import numpy as np
from urlextract import URLExtract
from statistics import mean,median,stdev

from scipy.stats import kurtosis
from scipy.stats import skew


import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
import matplotlib.pyplot as plt

from ImgurComments import Countries,galeries
from LoadTextData import Load_GalLery_Textual_Data

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

def split_count(text):
    emoji_counter = 0
    Symbols_counter =  0
    
    data = regex.findall(r'\X', text)
    
    extractor = URLExtract()
    urls_counter = len(extractor.find_urls(text))
    
    Mentions_counter = len(re.findall("@([a-zA-Z0-9]+)", text))
      
    #Symbols_counter = len([ch for word in text.split() for ch in word if not ch.isalnum()])
    
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            text = text.replace(word, '') 
        if any(not char.isalnum() for char in word):
           Symbols_counter += 1
    words_counter = len(text.split())
    
    return emoji_counter, words_counter,urls_counter,Mentions_counter,Symbols_counter

def Statistique():
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)

    Countries_Comments = {}
    Comments_word_Nb = {}
    #Comments_char_Nb = {}
    Countries_emogi = {}
    Countries_URLS = {}
    Countries_Mentions = {}
    Countries_Symbols = {}

    i = 0
    for Country in Countries:
        NB_Comments = []
        NB_W_Comments = []
        NB_emogi = []
        NB_URLS = []
        NB_Mentions = []
        NB_Symbols = []
        
        print(str(i+1) + ' : ' + Country)
        
        for j in range (10):
            Comments,Data = Load_GalLery_Textual_Data(Country, Galeries_Matrix[i,j])
            NB_Comments.append(len(Comments))
            for Comment in Comments:
                emoji_counter, words_counter,urls_counter,Mentions_counter,Symbols_counter = split_count(Comment)
                
                NB_W_Comments.append(words_counter)
                NB_emogi.append(emoji_counter)
                
                NB_URLS.append(urls_counter)
                NB_Mentions.append(Mentions_counter)
                NB_Symbols.append(Symbols_counter)
                
        Comments_word_Nb[Country] = NB_W_Comments
        Countries_Comments[Country] = NB_Comments
        Countries_emogi[Country] = NB_emogi
        
        Countries_URLS[Country] = NB_URLS
        Countries_Mentions[Country] = NB_Mentions
        Countries_Symbols[Country] = NB_Symbols
        
        i+=1
    return Countries_Comments,Comments_word_Nb,Countries_emogi,Countries_URLS,Countries_Mentions,Countries_Symbols
        
def Statistic_Measures(Dictinary):
    flat_list = [item for sublist in Dictinary.values() for item in sublist]
    Total = sum(flat_list)

    Avg_galery = [round(mean(Dictinary[key]),2) for key in Dictinary.keys()]
    Global_Avg_galery = round(mean(Avg_galery),2)

    stdev_galery = [round(stdev(Dictinary[key]),2) for key in Dictinary.keys()]
    Global_stdev_galery = round(mean(stdev_galery),2)

    data = flat_list
    print("Total : ",Total)
    print("Average : ",Global_Avg_galery)
    print("Stdev : ",Global_stdev_galery)
    print("skew : ",round(skew(data),2))
    print("kurt : ",round(kurtosis(data),2))
    print('______________________________')

#______________________________________________________________________________

Countries_Comments,Comments_word_Nb,Countries_emogi,Countries_URLS,Countries_Mentions,Countries_Symbols = Statistique()

#____________________________Coments Number____________________________________

Statistic_Measures(Countries_Comments)
Statistic_Measures(Comments_word_Nb)
Statistic_Measures(Countries_emogi)
Statistic_Measures(Countries_URLS)
Statistic_Measures(Countries_Mentions)
Statistic_Measures(Countries_Symbols)
