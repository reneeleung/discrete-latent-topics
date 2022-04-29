#!/usr/bin/env python
# coding: utf-8

import os
import gensim.corpora as corpora
import argparse
import logging
import time
import yaml
import tomotopy as tp
from utils import *
from models import HDP
from dataset import DocDataset
from multiprocessing import cpu_count


parser = argparse.ArgumentParser('HDP topic model')
parser.add_argument('--config',type=str,default=None,help='Config YAML file containing data paths and hyperparameters')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_iters',type=int,default=8000,help='Number of iterations (1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=5,help='Num of topics')
parser.add_argument('--term_weight',type=str,default='IDF',help='Term Weight for HDP model')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--ngrams',type=bool,default=True,help='Whether to include bigrams and trigrams in dictionary')
parser.add_argument('--rebuild',type=bool,default=False,help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default True)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')

args = parser.parse_args()

def main():
    global args
    
    config = args.config
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_iters = args.num_iters
    n_topic = args.n_topic
    term_weight = args.term_weight
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    ngrams = args.ngrams
    rebuild = args.rebuild
    auto_adj = args.auto_adj

    txtpath = None
    stopwords = None
    if config:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
            taskname = config['taskname']
            txtpath = config['txtpath']
            stopwords = config['stopwords']
            n_topic = config['n_topic']
            term_weight = config['term_weight']
            num_iters = config['num_iters'] if 'num_iters' in config else num_iters
            no_above = config['no_above'] if 'no_above' in config else no_above
            gamma = config['gamma'] if 'gamma' in config else 0.5
            rm_top = config['rm_top'] if 'rm_top' in config else 0

    docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams,use_tfidf=False)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams,use_tfidf=False)
    
    taskname = f'hdp_{taskname}'
    model_name = 'HDP'
    run_name= '{}_K{}_{}'.format(model_name,n_topic,taskname)
    makedir('logs')
    makedir('ckpt')
    loghandler = [logging.FileHandler(filename=f'logs/{run_name}.log',encoding="utf-8")]
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',handlers=loghandler)
    logger = logging.getLogger(__name__)


    if bkpt_continue:
        print('loading model ckpt ...')
        hdp_model = HDP.from_ckpt(hdp=tp.HDPModel.load('ckpt/{}.model'.format(run_name)))
    else:
        term_weight = {'IDF': tp.TermWeight.IDF,
                       'ONE': tp.TermWeight.ONE,
                       'PMI': tp.TermWeight.PMI}[term_weight]
        hdp_model = HDP(tw=term_weight,initial_k=n_topic,gamma=gamma,alpha=0.1,rm_top=rm_top)


    # Training
    print('Start Training ...')
    hdp_model.train(docSet.docs, num_iters)

    save_name = f'./ckpt/HDP_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    #hdp_model.hdp.save(save_name)


    # Evaluation
    print('Evaluation ...')
    topics_dict = hdp_model.get_topic_words()
    topic_words = [[x[0] for x in topic] for topic in topics_dict.values()]
    
    # Build Gensim vocab and corpus
    dictionary = corpora.Dictionary(docSet.docs)

    (cv_score, w2v_score, c_uci_score, c_npmi_score),_ = calc_topic_coherence(topic_words,docs=docSet.docs,dictionary=dictionary,taskname=taskname)

    topic_diversity = calc_topic_diversity(topic_words)

    result_dict = {'cv':cv_score,'w2v':w2v_score,'c_uci':c_uci_score,'c_npmi':c_npmi_score}
    logger.info('Topics:')

    save_directory = f'results/{taskname}'
    makedir(save_directory)

    with open(f'{save_directory}/{taskname}.txt', 'w') as f:
        for idx, vals in topics_dict.items():
            words = [x[0] for x in vals]
            logger.info(f'##{idx:>3d}:{words}')
            print(f'##{idx:>3d}:{words}')
            f.write(f'##{idx:>3d}:{words}\n')

        for measure,score in result_dict.items():
            logger.info(f'{measure} score: {score}')
            print(f'{measure} score: {score}')
            f.write(f'{measure} score: {score}\n')

        logger.info(f'topic diversity: {topic_diversity}')
        print(f'topic diversity: {topic_diversity}')
        f.write(f'topic diversity: {topic_diversity}\n')
        # Inspect topic distribution in corpus
        # Sanity check: make sure no correlation between topic and order of document
        topic_dist = hdp_model.get_topics_in_corpus()
        topics, topic_dist = show_topic_dist_in_corpus(topic_dist, len(topics_dict))
        print(f'Topic distribution:\n{topic_dist}')
        f.write(f'Topic distribution:\n{topic_dist}')
        f.close()

        # Create word cloud images
        create_wordclouds(topics, topics_dict, save_directory)


if __name__ == '__main__':
    main()
