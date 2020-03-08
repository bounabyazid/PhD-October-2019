#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:25:14 2019
@author: polo
"""

'https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/'

import re

# spacy for lemmatization
import spacy

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# NLTK Stop words
from nltk.corpus import stopwords

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
          'beef curtains', 'bellend', 'bint', 'bitch', 'bloodclaat', 'bloody', 'bollocks', 'bugger', 
          'bullshit', 'child-fucker', 'christ on a bike', 'christ on a cracker', 'clunge', 'cock',  
          'crap', 'cunt', 'damn', 'dick', 'dickhead', 'fanny', 'feck', 'flaps', 'frigger', 'fuck', 'gash', 
          'ginger', 'git', 'goddam', 'goddamn', 'godsdamn', 'hell', 'holy shit', 'jesus', 'jesus christ',
          'jesus h. christ', 'jesus harold christ', 'jesus wept', 'jesus, mary and joseph', 'judas priest', 
          'knob', 'minge', 'minger', 'motherfucker', 'munter', 'nigga', 'nigger', 'pissed', 'pissed off', 
          'prick', 'punani', 'pussy', 'shit', 'shit ass', 'shitass', 'snatch', 'sod-off', 'son of a bitch', 
          'son of a motherless goat', 'son of a whore', 'sweet jesus', 'tits ', 'twat ', 'twatanus']
#______________________________________________________________________________

def Preprocessing(text):
    newText = text.lower()
    # Remove Urls
    newText = re.sub(r'http\S+', '', newText)
    # Remove Emails
    newText = re.sub('\S*@\S*\s?', '', newText)
    
    for Swear in Swears:
        newText = newText.replace(Swear, ' ')
        
    # Remove new line characters
    newText = re.sub('\s+', ' ', newText)
    # Remove distracting single quotes
    newText = re.sub("\'", "", newText)
    
    return newText

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word)>2] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en', disable=['parser', 'ner'])

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#______________________________________________________________________________

def PrepareData(DocList):

    data = [Preprocessing(sent) for sent in DocList]

    data = remove_stopwords(data)

    data_words = list(sent_to_words(data))

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    return data_lemmatized
 
def LDA(data_lemmatized,num_topics):
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    
    return lda_model,id2word,corpus

def Topics_Words(lda_model,num_words):
    Topic_Words = []
    #for index, topic in lda_model.show_topics(formatted=False, num_words= 30):
        #print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
    for index, topic in lda_model.show_topics(formatted=False, num_words= num_words):
        Topic_Words.append([w[0] for w in topic]) 
    return Topic_Words    
        
    
def Perplexity_Coherence(lda_model,id2word,corpus,data_lemmatized):
    # Compute Perplexity
    Perplexity = lda_model.log_perplexity(corpus)# a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    return Perplexity,coherence_lda