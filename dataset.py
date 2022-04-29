#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import gensim
import json
import pickle
import random
import torch
from tqdm import tqdm
from tokenization import SpacyTokenizer
from torch.utils.data import Dataset,DataLoader
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from utils import makedir
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

class DocDataset(Dataset):
    def __init__(self,taskname,txtpath=None,tokenizer=None,stopwords=None,no_below=5,no_above=0.1,hasLable=False,rebuild=False,ngrams=True,use_tfidf=False):
        cwd = os.getcwd()
        txtpath = os.path.join(cwd,'data',f'{taskname}.txt') if txtpath==None else txtpath
        tmp_dir = os.path.join(cwd,'data',taskname)
        self.texts = load_data(txtpath)
        self.dictionary = None
        self.bows,self.docs = None,None
        self.ngrams = ngrams
        self.use_tfidf = use_tfidf
        self.tfidf,self.tfidf_model = None,None
        makedir(tmp_dir)
        if not rebuild and os.path.exists(os.path.join(tmp_dir,'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmp_dir,'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmp_dir,'tfidf.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmp_dir,'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmp_dir,'docs.pkl'),'rb'))
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()} # because id2token is empty be default, it is a bug.
        else:
            print('Tokenizing ...')
            tokenizer = SpacyTokenizer(additional_stopwords=stopwords)
            self.docs = tokenizer.tokenize(self.texts)
            if self.ngrams:
                # bigrams and trigrams
                bigram_phrases = gensim.models.Phrases(
                    self.docs,
                    min_count=5, # phrase must appear at least 5 times
                    threshold=50 # larger threshold will give fewer phrases
                )
                trigram_phrases = gensim.models.Phrases(
                    bigram_phrases[self.docs],
                    threshold=50 # larger threshold will give fewer phrases
                )
                bigram = gensim.models.phrases.Phraser(bigram_phrases)
                trigram = gensim.models.phrases.Phraser(trigram_phrases)
                data_bigrams = [bigram[doc] for doc in self.docs]
                data_bigrams_trigrams = [trigram[bigram[doc]] for doc in data_bigrams]
                self.docs = data_bigrams_trigrams
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            #self.dictionary.filter_n_most_frequent(remove_n=20)
            #self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)  # use Dictionary to remove un-relevant tokens
            self.dictionary.compactify()
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()} # because id2token is empty by default, it is a bug.
            # convert to BOW representation
            self.bows, _docs = [],[]
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow!=[]:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
            self.docs = _docs
            if self.use_tfidf==True:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
            # serialize the dictionary
            gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir,'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmp_dir,'dict.txt'))
            pickle.dump(self.docs,open(os.path.join(tmp_dir,'docs.pkl'),'wb'))
            if self.use_tfidf:
                gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir,'tfidf.mm'),self.tfidf)
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')
        
    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs
    
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self,topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.dfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(dfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ndoc} = {(1.0*freq/ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self,topk=20):
        ntokens = sum([v for k,v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.cfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(cfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ntokens} = {(1.0*freq/ntokens):>.13f}')
    
    def topk_dfs(self,topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0*dfs_topk[-1][-1]/ndoc

class TestData(Dataset):
    def __init__(self, dictionary=None, txtpath=None, tokenizer=None,stopwords=None,no_below=5,no_above=0.1,ngrams=True,use_tfidf=False):
        cwd = os.getcwd()
        self.texts = load_data(txtpath)
        self.dictionary = dictionary
        self.bows,self.docs = None,None
        self.ngrams = ngrams
        self.use_tfidf = use_tfidf
        self.tfidf,self.tfidf_model = None,None
        # self.texts is the list of string, without any preprocessing.
        print('Tokenizing ...')
        tokenizer = SpacyTokenizer(additional_stopwords=stopwords)
        self.docs = tokenizer.tokenize(self.texts)
        if self.ngrams:
                # bigrams and trigrams
                bigram_phrases = gensim.models.Phrases(
                    self.docs,
                    min_count=5, # phrase must appear at least 5 times
                    threshold=50 # larger threshold will give fewer phrases
                )
                trigram_phrases = gensim.models.Phrases(
                    bigram_phrases[self.docs],
                    threshold=50 # larger threshold will give fewer phrases
                )
                bigram = gensim.models.phrases.Phraser(bigram_phrases)
                trigram = gensim.models.phrases.Phraser(trigram_phrases)
                data_bigrams = [bigram[doc] for doc in self.docs]
                data_bigrams_trigrams = [trigram[bigram[doc]] for doc in data_bigrams]
                self.docs = data_bigrams_trigrams
        # convert to BOW representation
        self.bows, _docs = [],[]
        for doc in self.docs:
            if doc is not None:
                _bow = self.dictionary.doc2bow(doc)
                if _bow!=[]:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
                else:
                    _docs.append(None)
                    self.bows.append(None)
            else:
                _docs.append(None)
                self.bows.append(None)
        self.docs = _docs
        if self.use_tfidf==True:
            self.tfidf_model = TfidfModel(self.bows)
            self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')

    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs

    def __iter__(self):
        for doc in self.docs:
            yield doc

if __name__ == '__main__':
    docSet = DocDataset('zhdd',rebuild=True)
    dataloader = DataLoader(docSet,batch_size=64,shuffle=True,num_workers=4,collate_fn=docSet.collate_fn)
    print('docSet.docs[10]:',docSet.docs[10])
    print(next(iter(dataloader)))
    print('The top 20 tokens in document frequency:')
    docSet.show_dfs_topk()
    print('The top 20 tokens in collections frequency:')
    input("Press any key ...")
    docSet.show_cfs_topk()
    input("Press any key ...")
    for doc in docSet:
        print(doc)
        break
    print(docSet.topk_dfs(20))
    
