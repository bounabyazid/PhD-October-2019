#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:25:14 2019
@author: polo
"""

'https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/'

import re
import json
import truecase

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from ImgurComments import Countries,galeries

import spacy
from gensim.utils import simple_preprocess

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# NLTK Stop words
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import pos_tag, word_tokenize
from nltk.tag import StanfordNERTagger

from YazidPreprocessing import URL_Removal, give_emoji_free_text, Remove_punctuations

from LoadTextData import Load_GalLery_Textual_Data,Load_GoogleVision_Labels

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
          
Swears = ['arse', 'arsehole', 'ass', 'ass-hat', 'ass-jabber', 'ass-pirate', 'assbag', 'assbandit', 
          'assbanger', 'assbite', 'assclown', 'asscock', 'asscracker', 'asses', 'assface', 'assfuck', 
          'assfucker', 'assgoblin', 'asshat', 'asshead', 'asshole', 'asshopper', 'assjacker', 'asslick', 
          'asslicker', 'assmonkey', 'assmunch', 'assmuncher', 'assnigger', 'asspirate', 'assshit', 
          'assshole', 'asssucker', 'asswad', 'asswipe', 'axwound', 'balls', 'bastard', 'beaver', 
          'beef curtains', 'bellend', 'bint', 'bitch', 'bloodclaat', 'bloody', 'bollocks', 'bugger', 'bastard',
          'bullshit', 'child-fucker', 'christ on a bike', 'christ on a cracker', 'clunge', 'cock',  
          'crap', 'cunt', 'damn', 'dick', 'dickhead', 'fanny', 'feck', 'flaps', 'frigger', 'fuck', 'gash', 
          'ginger', 'git', 'goddam', 'goddamn', 'godsdamn', 'hell', 'holy shit', 'jesus', 'jesus christ',
          'jesus h. christ', 'jesus harold christ', 'jesus wept', 'jesus, mary and joseph', 'judas priest', 
          'knob', 'minge', 'minger', 'motherfucker', 'munter', 'nigga', 'nigger', 'pissed', 'pissed off', 
          'prick', 'punani', 'pussy', 'shit', 'shit ass', 'shitass', 'snatch', 'sod-off', 'son of a bitch', 
          'son of a motherless goat', 'son of a whore', 'sweet jesus', 'tits ', 'twat ', 'twatanus']

DataSet = '/home/polo/.config/spyder-py3/PhD/PhD October 2019/Tourism48'

#______________________________________________________________________________

def Preprocessing(Text):
    newText = URL_Removal(Text)
    newText = give_emoji_free_text(newText)
    newText = Remove_punctuations(newText)
    # Remove Emails
        
    for Swear in Swears:
        newText = newText.replace(Swear, ' ')
        
    # Remove new line characters
    newText = re.sub('\s+', ' ', newText)
    # Remove distracting single quotes
    newText = re.sub("\'", "", newText)
    newText = re.sub("\"", "", newText)
    newText.strip()
    
    return newText

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word)>2] for doc in texts]
#______________________________________________________________________________

def NE_Tagger(Text):
    st = StanfordNERTagger('/home/polo/Downloads/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
					       '/home/polo/Downloads/stanford-ner-2018-02-27/stanford-ner.jar', encoding='utf-8')
    tokenized_text = word_tokenize(Text)
    classified_text = st.tag(tokenized_text)
    return classified_text
#_______________________________________________________________
    
def find_proper_nouns(Tagged_Text):
    proper_nouns = []
    i = 0
    while i < len(Tagged_Text):
         if Tagged_Text[i][1] == 'NNP':
            if Tagged_Text[i+1][1] == 'NNP':
               proper_nouns.append(Tagged_Text[i][0].lower()+" " +Tagged_Text[i+1][0].lower())
               i+=1
            else:
                proper_nouns.append(Tagged_Text[i][0].lower())
         i+=1
    return proper_nouns

#_______________________________________________________________

def get_entities(DocList):
    ListDoc = [Preprocessing(truecase.get_true_case(sent)) for sent in DocList]
    'https://www.geeksforgeeks.org/python-named-entity-recognition-ner-using-spacy/'
    
    black_list = ['CARDINAL','ORDINAL','DATE','ORG']
    
    nlp = spacy.load('en_core_web_sm')
    Entities = [[ent.text.lower() for ent in nlp(doc).ents if ent.label_ not in black_list and len(ent.text)>2] for doc in ListDoc]
    #Entities = [[(ent.text, ent.label_) for ent in nlp(doc).ents if ent.label_ not in black_list and len(ent.text)>2] for doc in ListDoc]
    Entities = list(set([' '.join(List) for List in remove_stopwords(Entities) if len(List)>0]))
    
    return Entities

#_______________________________________________________________
    
def FuzzyWazzy_SimilarityOverAll(Country,gallery_id):
    S ,Data  = Load_GalLery_Textual_Data(Country,gallery_id)
    labels ,Data1  = Load_GoogleVision_Labels(Country,gallery_id)
        
    setA = list(set([x.lower() for x in labels]))
    setB = get_entities(S)
    
    if len(setB) == 0:
       return 0.0
    
    overlap = 0
    
    for l in setA:
        for w in setB:
            if fuzz.ratio(l, w) >= 75:
               overlap += 1
    
    Similarity = round(float(overlap) / len(setA) * 100., 2)
               
#    print ('overlap = ',overlap)
#    print ('Labels = ',len(setA))
#    print ('Comments = ',len(setB))
#    print ('overlap(Labels,Comments)/Labels = ',Similarity)
        
    return Similarity
#_______________________________________________________________

def OverAll_Text_Similarity_DataSet():
   
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)

    i = 0
    for Country in Countries:
        print ('============='+Country+'==============')
        Slabels = []
        Similarities = {}
        for j in range (10):
            #print(Galeries_Matrix[i,j])
            Similarity = FuzzyWazzy_SimilarityOverAll(Country,Galeries_Matrix[i,j])
            Slabels.append(Similarity)
            
        Similarities['labels'] = Slabels
        
        with open('Similarities NER/'+Country+'.json', 'w') as outfile:
             json.dump(Similarities, outfile)
        #break             
        i+=1

def Histogramme(Country):
    with open('Similarities NER/'+Country+'.json') as data_file:    
         Data = json.load(data_file)
    #plt.hist(Data['overall'])
    
    x = np.arange(10)
    plt.bar(x, Data['labels'])
    plt.xticks(x+.2, x)
    
def Globalmean():
    means = []
    for Country in Countries:
        #print ('============='+Country+'==============')
        with open('Similarities NER/'+Country+'.json') as data_file:    
             Data = json.load(data_file)
        means.append(round(mean(Data['labels']), 2))

    x = np.arange(48)
    plt.bar(x, means)
    plt.xticks(x+.2, x)
    
def GlobalHistogram():
    means = []
    for Country in Countries:
        #print ('============='+Country+'==============')
        with open('Similarities NER/'+Country+'.json') as data_file:    
             Data = json.load(data_file)
        means.append(Data['labels'])
        
    meanGaleries = np.mean(means, axis=0)
    print(meanGaleries)
    x = np.arange(10)
    plt.bar(x, meanGaleries)
    plt.xticks(x+.2, x)

#FuzzyWazzy_SimilarityOverAll('Algeria','6aCY1be')
#OverAll_Text_Similarity_DataSet()

#S ,Data  = Load_GalLery_Textual_Data('Algeria','6aCY1be')
#labels ,Data1  = Load_GoogleVision_Labels('Algeria','6aCY1be')

#Similarity = FuzzyWazzy_SimilarityOverAll('Cambodia','m2TvHjv')
#S ,Data  = Load_GalLery_Textual_Data('Cambodia','m2TvHjv')
#labels ,Data1  = Load_GoogleVision_Labels('Cambodia','m2TvHjv')
#
#setA = list(set([x.lower() for x in labels]))
#setB = get_entities(S)


#Histogramme('France')

#Globalmean()
GlobalHistogram()
