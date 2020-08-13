import os
import re
import math
import time
from tqdm.auto import tqdm
import timeit
import pickle
import logging
from collections import Counter
from copy import deepcopy,copy
import numpy as np
# from memory_profiler import profile

import torch
# import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,BatchSampler, WeightedRandomSampler
from torch.optim import Adam, AdamW
from scipy.stats import pearsonr, spearmanr

from src.optim import get_cosine_schedule_with_warmup
from src.model import NCEloss
from src.eval import cosine_sim_rg, cosine_sim_sim393

SEED = 42 # predictable random variables
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
logger =logging.getLogger(__name__)

SAVE_DIR = './result_0812'
rm_words = ['\n','=']
window_size = 5
t = 1e-05

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
else:
    pass

class Word2VecDataset(Dataset):
    """ data to array, tensorize """
    def __init__(self, data_pth=None, data=None):
        super().__init__()
        if data is None:
            self.data = pickle.load(open(data_pth, 'rb'))
        else:
            self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iword, owords = self.data[idx][0], self.data[idx][1:]
        return torch.LongTensor([iword]), torch.LongTensor(owords)

    
def my_collate(batch):
    length = torch.tensor([ torch.tensor(words[1].shape[0]) for words in batch])
    batch_iwords = torch.LongTensor([words[0] for words in batch])
    batch_owords = [torch.LongTensor(words[1]) for words in batch]
    batch_owords = torch.nn.utils.rnn.pad_sequence(batch_owords)
    ## compute mask
    #mask = (batch_owords != 0).to(device)
    return batch_iwords, batch_owords

# Hyper-params
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu' # same as torch.device(0)
# model
NAME = 'skip_gram'
BATCH_SIZE = 512
EPOCHS = 5
HIDDEN = 100
lr = 0.001
key = time.asctime().split()[3][:2]

# with open('./result_0812/data.pkl', 'rb') as r:
#     data = pickle.load(r)
# logger.info("data loaded")

with open('./result_0810/vocab.pkl', 'rb') as r:
    word2idx, idx2word = pickle.load(r)
logger.info("vocab loaded")

with open('./result_0810/featrues.pkl', 'rb') as r:
    features = pickle.load(r)
logger.info("featrues loaded")   

with open('./result_0810/unigram.pkl', 'rb') as r:
    unigram, denominator = pickle.load(r)

# Evaluation Files
path_rg65 = './dataset/wrd/eval/RG65.csv'
path_sim393 = './dataset/wrd/eval/wordsim353_sim_rel/wordsim353_annotator1.txt'

with open(path_rg65, 'r') as reader:
    rg65 = reader.readlines()

sim393 = open(path_sim393, 'r').readlines()

# sub sampling 수정 => 데이터셋을 만들 때, 아예 제외시키고 윈도우 데이터셋을 구성
#dataset = Word2VecDataset('./util/features0728.pkl')
dataset = Word2VecDataset(data=features)
logger.info(len(dataset) // BATCH_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)


t_total = len(dataloader) * BATCH_SIZE # total step
vocab_size = len(word2idx)
nce = NCEloss(vocab_size, HIDDEN, neg_n=5,distrib=unigram,padding_idx=0)
nce = nce.to(device)
optimizer = Adam(nce.parameters(),lr=lr)
scheduler = get_cosine_schedule_with_warmup(optimizer,0,t_total)

best_loss = np.inf
best_rg = -np.inf
best_393 = -np.inf
global_step = 0
for epoch in range(EPOCHS):
    pbar =  tqdm(dataloader)
    train_loss = 0.0
    step=0
    for iword, owords in pbar:
        iword, owords = iword.to(device), owords.to(device)
        loss = nce(iword, owords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        train_loss += loss.item()
        step+=1
        global_step+=1

    logger.info(f"Epoch[{epoch}/{EPOCHS}] Train Loss[{float(train_loss / step):.4f}] Learning rate: {optimizer.param_groups[0]['lr']}")
    
    if global_step % 20000 == 0:
        torch.save(nce.state_dict(), os.path.join(SAVE_DIR, '{}_{}_{}_{}_{}.pt'.format(NAME,HIDDEN,lr,key,global_step)))
        logger.info("step measure save model...")
            
    with torch.no_grad():
        logger.info("Evaluate Model")
        nce.eval()
        sim_lst_rg, sim_human = cosine_sim_rg(rg65, nce, word2idx)
        r_rg = spearmanr(sim_lst_rg,sim_human).correlation
        sim_lst_393, sim_human = cosine_sim_sim393(sim393, nce, word2idx)
        r_393 = spearmanr(sim_lst_393,sim_human).correlation
        
    logger.info(f"r_rg :{r_rg} r_393:{r_393}")
        
    if best_loss > train_loss and r_rg > best_rg and r_393 > best_393:
        best_rg = r_rg
        best_393 = r_393
        best_loss = train_loss
        torch.save(nce.state_dict(), os.path.join(SAVE_DIR, '{}_{}_{}_{}.pt'.format(NAME,HIDDEN,lr,key)))
        logger.info("Model saved...")