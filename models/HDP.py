"""
Wrapper class for tomotopy HDPModel to perform various functions
"""
import tomotopy as tp
from multiprocessing import cpu_count
import numpy as np
import os

n_cpu = cpu_count()-2 if cpu_count()>2 else 2

class HDP:
    def __init__(self, tw, initial_k, gamma, alpha, min_cf=5, rm_top=0, hdp=None):
        if hdp:
            self.hdp = hdp
        else:
            self.hdp = tp.HDPModel(tw=tw,initial_k=initial_k,gamma=gamma,alpha=alpha,min_cf=min_cf,rm_top=rm_top,seed=88888)

    @classmethod
    def from_ckpt(cls, hdp):
        cls(None,None,None,None,None,None,hdp)

    def train(self, docs, iters, burn_in=100):
        for doc in docs:
            self.hdp.add_doc(doc)

        # MCMC - discard first burn_in samples
        self.hdp.burn_in = burn_in
        self.hdp.train(0)
        print(f'Number of docs: {len(self.hdp.docs)}, Vocab size: {self.hdp.num_vocabs}, Number of words: {self.hdp.num_words}')
        print(f'Removed top words: {self.hdp.removed_top_words}')

        step = round(iters*0.01)
        for i in range(0, iters, step):
            self.hdp.train(step, workers=n_cpu)
            print(f'Iteration: {i}\tLog-likelihood: {self.hdp.ll_per_word}\tNo. of topics: {self.hdp.live_k}')

    def get_topic_words(self, topK=15):
        # Get most important topics by # of times they were assigned (i.e. counts)
        sorted_topics = [k for k, v in sorted(enumerate(self.hdp.get_count_by_topics()), key=lambda x:x[1], reverse=True)]

        topics = {}
        # For topics found, extract only those that are still assigned
        for k in sorted_topics:
            if not self.hdp.is_live_topic(k): continue # remove un-assigned topics at the end (i.e. not alive)
            topic_wp = []
            for word, prob in self.hdp.get_topic_words(k, top_n=topK):
                topic_wp.append((word, prob))

            topics[k] = topic_wp # store topic word/frequency array (replace topic #)
        return topics

    def get_topics_in_corpus(self):
        return [self.get_dominant_topic(doc) for doc in self.hdp.docs]

    def get_dominant_topic(self, doc):
        topic_dist, _ = self.hdp.infer(doc) # note this will have shape including topics that are not "alive"
        topic_idx = np.array(topic_dist).argmax()
        return topic_idx

if __name__ == '__main__':
    model = HDP(tp.TermWeight.ONE, 5, 1, 0.1)
