#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:56:15 2019
@author: polo
"""  

import json

import numpy as np
import matplotlib.pyplot as plt

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from gensim.summarization import keywords

from LoadTextData import Load_GalLery_Textual_Data,Load_GoogleVision_Labels
from LDA import Preprocessing,remove_stopwords,sent_to_words,lemmatization,PrepareData,LDA,Topics_Words

from ImgurComments import Countries,galeries
    
DataSet = '/home/polo/.config/spyder-py3/PhD/Tourism30'

def LoadTextData(Country,gallery_id):
    S ,Data  = Load_GalLery_Textual_Data(Country,gallery_id)
    
    S1 ,Data1  = Load_GoogleVision_Labels(Country,gallery_id)
    
    labels = [Preprocessing(x['label']) for x in S1[0]]
    labels.append(Preprocessing(S1[1]))
    
    DocList = S[1]
    DocList.append(S[0])

    for s in S[2]:
        DocList.extend(s)
    
    data_lemmatized = PrepareData(DocList)
    lda_model,id2word,corpus = LDA(data_lemmatized,num_topics=20)#len(labels))
    Topic_Words = Topics_Words(lda_model,num_words=len(labels))   
    
    return Topic_Words,labels

def Algebric_Similarity(Country,gallery_id):
    Topics,labels = LoadTextData(Country,gallery_id)

    setA = set([x.lower() for x in labels])
    i = 0
    for Topic in Topics:
        print ('____________Topic:{}_____________'.format(i))
        setB = set(Topic)
        overlap = setA & setB
        universe = setA | setB

        result1 = float(len(overlap)) / len(setA) * 100
        result2 = float(len(overlap)) / len(setB) * 100
        result3 = float(len(overlap)) / len(universe) * 100
        
        print ('overlap = ',len(overlap))
        print ('universe = ',len(universe))

        print ('overlap(setA,setB)/setA = ',round(result1, 2))
        print ('overlap(setA,setB)/setB = ',round(result2, 2))
        print ('overlap(setA,setB)/universe(setA,setB) = ',round(result3, 2))
        
        i = i+1
      
def FuzzyWazzy_Similarity(Country,gallery_id):
    'https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/'
    Topics,labels = LoadTextData(Country,gallery_id)

    setA = set([x.lower() for x in labels])
    i = 0
    for Topic in Topics:
        print ('____________Topic:{}_____________'.format(i))
        setB = set(Topic)
        
        overlap = 0
        for l in setA:
            for w in setB:
                if fuzz.ratio(l, w) >= 80:
                   overlap += 1
       
        universe = setA | setB

        result1 = float(overlap) / len(setA) * 100
        result2 = float(overlap) / len(setB) * 100
        result3 = float(overlap) / len(universe) * 100
        
        print ('overlap = ',overlap)
        print ('universe = ',len(universe))
        
        print ('overlap(setA,setB)/setA = ',round(result1, 2))
        print ('overlap(setA,setB)/setB = ',round(result2, 2))
        print ('overlap(setA,setB)/universe(setA,setB) = ',round(result3, 2))
        i = i+1



def FuzzyWazzy_SimilarityOverAll(Country,gallery_id):
    Topics,labels = LoadTextData(Country,gallery_id)

    #print ('=============OverAll Similarity==============')
    
    setA = list(set([x.lower() for x in labels]))
    setB = list(set([w for Topic in Topics for w in Topic]))
    
    overlap = 0
    universe = 0
    
    for l in setA:
        for w in setB:
            if fuzz.ratio(l, w) >= 80:
               overlap += 1
            else:
                universe += 1

    labels = round(float(overlap) / len(setA) * 100., 2)
    comments = round(float(overlap) / len(setB) * 100., 2)
    overall = round(float(overlap) / float(universe) * 100., 2)
        
    #print ('overlap = ',overlap)
    #print ('universe = ',universe)
    
    #print ('Labels = ',len(setA))
    #print ('Comments = ',len(setB))

    #print ('overlap(Labels,Comments)/Labels = ',labels)
    #print ('overlap(Labels,Comments)/Comments = ',comments)
    
    print ('overlap(Labels,Comments)/Universe(Labels,Comments) = ',overall)
    
    return labels,comments,overall

def keyWords_Labels_Matching(Country,gallery_id):
    S ,Data  = Load_GalLery_Textual_Data(Country,gallery_id)
    S1 ,Data1  = Load_GoogleVision_Labels(Country,gallery_id)

    DocList = S[1]
    DocList.append(S[0])
    for s in S[2]:
        DocList.extend(s)
        
    data_lemmatized = [w for doc in PrepareData(DocList) for w in doc]
    
    print (data_lemmatized)
    
    fullStr = ' '.join(data_lemmatized)
    
    #labels = [Preprocessing(x['label']) for x in S1[0]]
    #labels.append(Preprocessing(S1[1]))

    labels = [w for label in PrepareData(S1) for w in label]
        
    setA = list(set(labels))
    
    setB = keywords(fullStr).split('\n')

    setB = [w for docs in PrepareData(setB) for w in docs]
  
    overlap = 0
    
    for l in setA:
        for w in setB:
            if fuzz.ratio(l, w) >= 75:
               overlap += 1
               
    universe = []
    
    uni = list(set(setA) | set(setB))
        
    for i in range(len(uni)):
        if uni[i] not in universe:
           universe.append(uni[i]) 
        for j in range(i+1,len(uni)):
            if fuzz.ratio(uni[i], uni[j]) >= 75 and uni[j] not in universe:
               universe.append(uni[j])
               
    universe = len(universe)
    
    labels = round(float(overlap) / len(setA) * 100., 2)
    comments = round(float(overlap) / len(setB) * 100., 2)
    overall = round(float(overlap) / float(universe) * 100., 2)
        
    #print ('overlap = ',overlap)
    #print ('universe = ',universe)
    
    #print ('\nLabels = ',len(setA))
    #print ('Comments = ',len(setB))

    #print ('overlap(Labels,Comments)/Labels = ',labels)
    #print ('overlap(Labels,Comments)/Comments = ',comments)
    
    print ('overlap(Labels,Comments)/Universe(Labels,Comments) = ',overall)
    
    
    return labels,comments,overall,setA,setB
        
def OverAll_Text_Similarity_DataSet():
   
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)

    i = 0
    for Country in Countries:
        print ('============='+Country+'==============')

        Slabels = []
        Scomments = []
        Soverall = []
        
        Similarities = {}
        
        for j in range (10):
            #print(Galeries_Matrix[i,j])
            #labels,comments,overall = FuzzyWazzy_SimilarityOverAll(Country,Galeries_Matrix[i,j])
            labels,comments,overall = keyWords_Labels_Matching(Country,Galeries_Matrix[i,j])
            
            Slabels.append(labels)
            Scomments.append(comments)
            Soverall.append(overall)
        
        Similarities['labels'] = Slabels
        Similarities['comments'] = Scomments
        Similarities['overall'] = Soverall
        
        with open(DataSet+'/'+Country+'/Similarities.json', 'w') as outfile:
             json.dump(Similarities, outfile)
        #break             
        i+=1

def Histogramme(Country):
    with open(DataSet+'/'+Country+'/Similarities.json') as data_file:    
         Data = json.load(data_file)
    #plt.hist(Data['overall'])
    
    x = np.arange(10)
    plt.bar(x, Data['overall'])
    plt.xticks(x+.2, x)

#OverAll_Text_Similarity_DataSet()
#Histogramme('Algeria')

labels,comments,overall,setA,setB = keyWords_Labels_Matching('Algeria','x6TwpSQ')