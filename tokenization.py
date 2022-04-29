import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords

from typing import List
import multiprocessing

import spacy
import gensim

class SpacyTokenizer(object):
    def __init__(self, additional_stopwords=None):
        self.stopwords = stopwords.words("english")
        if additional_stopwords:
            self.stopwords += additional_stopwords
        self.nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
        print("Using SpaCy tokenizer")

        
    def tokenize(self, texts: List[str], allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]) -> List[List[str]]:
        docs = self.nlp.pipe(texts, batch_size=1000, n_process=multiprocessing.cpu_count())
        docs = [" ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags]) for doc in docs]
        docs = [gensim.utils.simple_preprocess(doc, deacc=True) for doc in docs]
        docs = [[word for word in doc if word not in self.stopwords] for doc in docs]
        docs = list(filter(len, docs))
        return docs

 
