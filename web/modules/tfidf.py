#!/bin/env python
import gensim
import numpy as np
import difflib

import logging, gensim


class IterCorpus(object):
  
  def __init__(self, dictionary, corpus_array):
      self.dictionary = dictionary
      self.corpus_array = corpus_array
      
  def __iter__(self):
      for line in self.corpus_array:
          yield self.dictionary.doc2bow(line[0].split(' '))


class tfidf_kernel:

  def __init__(self, corpus_array):
      
      self.dictionary = gensim.corpora.Dictionary(line[0].replace(' barcelona','').split(' ') for line in corpus_array)
      

      self.corpus = [self.dictionary.doc2bow(line[0].replace(' barcelona','').split(' ')) for line in corpus_array] 
      #self.corpus = IterCorpus(self.dictionary, corpus_array) 
      
      
      self.tfidf = gensim.models.TfidfModel(self.corpus)
      
      #self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=3, update_every=0, passes=20)
      #self.lda_index = gensim.similarities.MatrixSimilarity(self.lda[self.corpus]) # transform corpus to LSI space and index it
 
      
  def eval_lsi(self, str_a, str_b):
      
      if len(str_a) == 0 or len(str_b) == 0:
        return 0
      
      a = self.dictionary.doc2bow(str_a.split(' '))
      b = self.dictionary.doc2bow(str_b.split(' '))
      
      vec_lda1 = self.lda[a]
      sims = self.lda_index[vec_lda1]
  
      indb = [i for i,x in enumerate(self.corpus) if x == b]
            
      sim = sims[indb][0] 
      
      return sim  


  def eval_tfidf(self, str_a, str_b):
      
      if len(str_a) == 0 or len(str_b) == 0:
        return 0
      
      a = self.dictionary.doc2bow(str_a.split(' '))
      b = self.dictionary.doc2bow(str_b.split(' '))

      tfidf1 = self.tfidf[a]
      tfidf2 = self.tfidf[b]

      index = gensim.similarities.MatrixSimilarity([tfidf1],num_features=len(self.dictionary))
      sim = index[tfidf2]
        
      return sim  
  
  
  def kernel_function_submatch(self, strings_A, strings_B):
      
      dd = np.zeros((len(strings_B),len(strings_A)))
      for iA, strA in enumerate(strings_A):
         for iB, strB in enumerate(strings_B):
           
           a = str(strA[0]).replace(' barcelona','')
           b = str(strB[0]).replace(' barcelona','')
           
           w = difflib.SequenceMatcher(None, a=a, b=b).ratio()
           
           gamma = 2
                      
           dd[iB][iA] = w #np.exp(-gamma*(1-w)*(1-w))
                       
      return dd


  def kernel_function_tfidf_slow(self, strings_A, strings_B):

      dd = np.zeros((len(strings_B),len(strings_A)))
      for iA, strA in enumerate(strings_A):
         for iB, strB in enumerate(strings_B):
           
           a = str(strA[0]).replace(' barcelona','')
           b = str(strB[0]).replace(' barcelona','')
           
           w = self.eval_tfidf(a, b)
                
           gamma = 2
                      
           dd[iB][iA] = w #np.exp(-gamma*(1-w)*(1-w))
     
      return dd
  
  
  def kernel_function_tfidf(self, strings_A, strings_B):

      tfidf1 = self.tfidf[[self.dictionary.doc2bow(str(str_a[0]).replace(' barcelona','').split(' ')) for str_a in strings_A]]
      tfidf2 = self.tfidf[[self.dictionary.doc2bow(str(str_b[0]).replace(' barcelona','').split(' ')) for str_b in strings_B]]
      index = gensim.similarities.MatrixSimilarity(tfidf1,num_features=len(self.dictionary))
      sim = index[tfidf2]
      
      return sim


  def kernel_function_lsi(self, strings_A, strings_B):
      
      dd = np.zeros((len(strings_B),len(strings_A)))
      for iA, strA in enumerate(strings_A):
         for iB, strB in enumerate(strings_B):
           
           a = str(strA[0]).replace(' barcelona','')
           b = str(strB[0]).replace(' barcelona','')
           
           w = self.eval_lsi(a, b)
                
           gamma = 2
                      
           dd[iB][iA] = w #np.exp(-gamma*(1-w)*(1-w))
                       
      return dd
    
