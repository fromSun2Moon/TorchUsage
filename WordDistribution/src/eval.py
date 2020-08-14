# -*- coding: utf-8 -*-
# Evaluation Model
import re
import os
import pickle
import logging
import numpy as np
import pandas as pd
from numpy.linalg import norm
#import matplotlib.pylab as plt

import torch
from src.model import NCEloss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

def custom_cosine_similarity(x, y)-> int: 
    """ 1D tensor input """
    return x.dot(y) / (norm(x)* norm(y) + 1e-15) # l2 norm in paper

def cosine_sim_men(men, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for word1, word2, num in men:
        w1 = word1.replace('-n','').replace('-j','').replace('-v','')
        w2 = word2.replace('-n','').replace('-j','').replace('-v','')
        try:
            x = model.embedding_i.weight[word2idx[w1]]
            y = model.embedding_i.weight[word2idx[w2]]
            sim = custom_cosine_similarity(x.detach().numpy(), y.detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(num.replace("\n","")))
        except Exception as e:
            print(str(e))
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human


def cosine_sim_rg(rg65, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []
    for i in range(len(rg65)):
        try:
            x = model.embedding_i.weight[word2idx[rg65[i].split(';')[0]]]
            y = model.embedding_i.weight[word2idx[rg65[i].split(';')[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(rg65[i].split(';')[2].replace(' ','').replace('\n','')))
        except Exception as e:
            rm_cnt+=1
            print(str(e))
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_sim393(sim393, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []

    for lst in sim393[11:]:
        line = lst.split('\t') 
        try:
            x = model.embedding_i.weight[word2idx[line[1]]]
            y = model.embedding_i.weight[word2idx[line[2]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(line[3].replace("\n","")))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_simrel(sim_rel, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []

    for lst in sim_rel:
        line = lst.split('\t') 
        try:
            x = model.embedding_i.weight[word2idx[line[0]]]
            y = model.embedding_i.weight[word2idx[line[1]]]
            sim = custom_cosine_similarity(x.detach().numpy(), y.detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(line[2].replace("\n","")))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_simlex(simlex, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for line in simlex:
        line = line.split('\t')
        try:
            x = model.embedding_i.weight[word2idx[line[0]]]
            y = model.embedding_i.weight[word2idx[line[1]]]
            sim = custom_cosine_similarity(x.detach().numpy(), y.detach().numpy())
            sim_lst.append(sim)
            sim_human.append(line[3])
        except Exception as e:
            print(e)
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human


if __name__ == '__main__':
    
    SEED = 42 # predictable random variables
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)
    
    # Model load
    ##########################################
    ##### Free available for Model format ####
    ##########################################
    model_path = 'result_0810'
    name = 'skip_gram'
    hidden = 100
    lr = 0.001
    key = '16'

    with open(f'./{model_path}/vocab.pkl', 'rb') as reader:
        vocab = pickle.load(reader)
        word2idx, idx2word = vocab
    
    vocab_size = len(vocab[0]) # 16.7M

    model = NCEloss(vocab_size, hidden)
    model.load_state_dict(torch.load(f'./{model_path}/{name}_{hidden}_{lr}_{key}.pt'))
    model.eval()
    logger.info("Model loaded")


