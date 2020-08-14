# -*- coding: utf-8 -*-
import os
import re
import sys
import math
import pickle
import logging
from tqdm.auto import tqdm
from collections import Counter
sys.path.append('./')
from utils import preprocess, rm, get_vocab

# global varialbes
window_size = 5
t = 1e-05
SAVE_DIR = './result'

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
else:
    pass


def get_unigram_distribution(words):
    # unigram distribution
    occurs_vocab = Counter(words) 
    denominator = sum(list(map(lambda x : x[1] ,occurs_vocab.items())))  #only python operation
    return list(map(lambda x : x[1] ,occurs_vocab.items())), denominator

def get_frequency(unigram, denominator):
    frequency = list(map(lambda x : x / denominator, unigram))
    return frequency

def get_over_threshold(frequency, t=1e-05):
    # count frequency greater than t
    return list(filter(lambda x : x > t, frequency))

def get_features(data, frequency):
    global t
    global SAVE_DIR
    global window_size
    features=[]
    for doc in tqdm(data):
        doc = doc.split(' ')
        try:
            while True:
                doc.remove('(')
                doc.remove(')')
                doc.remvoe('.')
        except:
            pass
        
        for idx in range(len(doc)):
            word= doc[idx]
            if frequency[word2idx[word]] < t:
                left = doc[max(idx-window_size,0):idx]
                right = doc[idx+1:idx+(window_size+1)]
                #features.append([word]+['<unk>' for _ in range(window_size - len(left))]+left + right+ ['<unk>' for _ in range(window_size - len(right))])
                features.append([word] + left + right)
            else:
                pass

    with open(os.path.join(SAVE_DIR,'data.pkl'), 'wb') as writer:
        pickle.dump(features,writer)
    logger.info("Data(words) saved....")

    indices = []
    for words in tqdm(features):
        try:
            indices.append([word2idx[word] for word in words])
        except Exception as e:
            print("Error : ", e)

    with open(os.path.join(SAVE_DIR,'featrues.pkl'), 'wb') as writer:
        pickle.dump(indices, writer)
    logger.info("Features saved....")
    return features, indices




if __name__ == '__main__':

    data_path = '../dataset/wrd/wikitext-2/wiki.train.tokens' # path for data

    # Logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)

    # test module
    with open(data_path, 'r') as reader:
        raw_data = reader.readlines()
    
    data = preprocess(raw_data)
    logger.info(f"doc's numbers : {len(data)}")

    data_ = list(map(rm, data))
    words, word2idx, idx2word = get_vocab(data_)
    unigram, denominator = get_unigram_distribution(words)
    logger.info(f"Word's numbers : {denominator}") 
    frequency = get_frequency(unigram, denominator)
    features = get_features(data_, frequency)

    del features
