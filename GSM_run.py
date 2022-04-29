#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Modified from
'''
@File    :   GSM_run.py
@Time    :   2020/09/30 15:52:35
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import torch
import pickle
import argparse
import logging
import time
import yaml
from models import GSM, GSB, RSB
from utils import *
from dataset import DocDataset
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('GSM/GSB/RSB topic model')
parser.add_argument('--config',type=str,default=None,help='Config YAML file containing data paths and hyperparameters')
parser.add_argument('--model_name',type=str,default='gsm',help='Use model gsm or gsb or rsb')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=1000,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=5,help='Num of topics')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--ngrams',type=bool,default=True,help='Whether to include bigrams and trigrams in dictionary')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--criterion',type=str,default='cross_entropy',help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--ckpt',type=str,default=None,help='Checkpoint path')

args = parser.parse_args()

def main():
    global args
    config = args.config
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    ngrams = args.ngrams
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    batch_size = args.batch_size
    criterion = args.criterion
    auto_adj = args.auto_adj
    ckpt = args.ckpt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txtpath = None
    stopwords = None
    if config:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
            taskname = config['taskname']
            txtpath = config['txtpath'] # JSON format
            stopwords = config['stopwords']
            model_name = config['model_name']
            num_epochs = config['num_epochs'] if 'num_epochs' in config else num_epochs
            n_topic = config['n_topic'] if 'n_topic' in config else n_topic
            # Runs n trials for number of topics ranging from lower to upper
            lower = config['lower'] if 'lower' in config else None
            upper = config['upper'] if 'upper' in config else None
            n_trials = config['n_trials'] if 'n_trials' in config else 1

    docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams,use_tfidf=False)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams,use_tfidf=False)
    
    taskname = f'{model_name}_{taskname}'
    voc_size = docSet.vocabsize
    print('voc size:',voc_size)

    if ckpt:
        checkpoint=torch.load(ckpt)
        param.update({"device": device})
        if model_name == 'gsm':
            model = GSM(**param)
        elif model_name == 'gsb':
            model = GSB(**param)
        elif model_name == 'rsb':
            model = RSB(**param)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion,ckpt=checkpoint)
    else:
        # Run experiment to find optimal no. of topics
        if not lower or not upper:
            lower, upper = n_topic, n_topic
        for i in range(n_trials):
            tp, cv, td = [], [], []
            for n_topic in range(lower, upper+1):

                if model_name == 'gsm':
                    model = GSM(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device)
                elif model_name == 'gsb':
                    model = GSB(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device)
                elif model_name == 'rsb':
                    model = RSB(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device)
                else:
                    raise Exception('Unknown model')

                save_directory = f'results/{taskname}'
                makedir(save_directory)
                model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion,save_directory=save_directory)
                cv_score, _, _, _, _, topic_diversity = model.evaluate(test_data=docSet, w2file=f'{save_directory}/{taskname}_tp{n_topic}.txt')
                
                topic_dist = model.get_topics_in_corpus(docSet)
                print(topic_dist)
                topics, topic_dist = show_topic_dist_in_corpus(topic_dist, topk=n_topic)
                print(f'Topic distribution: {topic_dist}')
                with open(f'{save_directory}/{taskname}_tp{n_topic}.txt', 'a') as f:
                    f.write(f'Topic distribution:\n{topic_dist}')
                    f.close()
                if lower == upper:
                    # Create word clouds
                    topics_dict = model.show_topic_words(topK=15, showWght=True)
                    create_wordclouds(topics, topics_dict, save_directory)

                
                cv.append(cv_score)
                td.append(topic_diversity)
                tp.append(n_topic)
                save_name = f'./ckpt/{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
                # Save some disk space
                """
                torch.save(model.vae.state_dict(),save_name)
                txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
                with open(f'topic_dist_{model_name}.txt','w',encoding='utf-8') as wfp:
                    for t,e in zip(txt_lst,embeds):
                        wfp.write(f'{e}:{t}\n')
                pickle.dump({'txts':txt_lst,'embeds':embeds},open(f'{model_name}_embeds.pkl','wb'))
                """
            if lower != upper:
                with open(f'{save_directory}/{model_name}_tp_{i}.pkl', 'wb') as f:
                    pickle.dump(tp, f)
                with open(f'{save_directory}/{model_name}_cv_{i}.pkl', 'wb') as f:
                    pickle.dump(cv, f)
                with open(f'{save_directory}/{model_name}_td_{i}.pkl', 'wb') as f:
                    pickle.dump(td, f)
                print(f'Pickle dumped at trial {i}, #topics {n_topic}')

if __name__ == "__main__":
    main()
