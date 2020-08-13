# -*- coding: utf-8 -*-
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,BatchSampler, WeightedRandomSampler

from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR 

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu' 

#############################################
         ###### RNNLM Model ########
#############################################


#############################################
         ###### CBOW Model ########
#############################################


#############################################
###### Skip-Gram Model with NCE loss ########
#############################################

class NCEloss(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_n=10, power=0.75, distrib=None, padding_idx=0):
        """ 
        Args : weights list(vocab_size)
        """
        super(NCEloss, self).__init__()
        self.neg_n = neg_n
        self.vocab_size = vocab_size
        self.embedding_i = nn.Embedding(vocab_size, embed_dim,padding_idx=0) # vocab size, hidden size
        self.embedding_os = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        
        self.embed_dim = embed_dim
        self.power = power
        self.distrib = distrib
        self.initialize()
        
    def initialize(self):
        """
        init the weight as original word2vec do.
        :return: None
        """
        initrange = 0.5 / self.embed_dim
        self.embedding_i.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_os.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_i.weight.requires_grad=True
        self.embedding_os.weight.requires_grad=True
                                                          
    def forward(self, i_word, o_words): # obj generate 후, call함.
        """
        i_word: in word 
        o_words: out words
        """
        batch_size = i_word.shape[0]
        context_size = o_words.shape[0]
        
        # paper, Negative sampling distribution : proposed distribution U(w)^3/4 /Z
        if self.distrib is not None:
            wt = torch.pow(torch.FloatTensor(self.distrib), self.power)
            wt /= wt.sum()
            wt = wt.to(device)
            # make uniform distribution changed 
            wt[i_word] = 0
            wt[o_words] = 0
            
        n_words = torch.multinomial(wt ,batch_size * context_size * self.neg_n, replacement=True).view(batch_size, -1)
        
        # weights uniform distribution
        if self.distrib is None:
            n_words = torch.empty(batch_size *context_size * self.neg_n).uniform_(0, self.vocab_size-1).long()
        
        # version : 
        o_words = o_words.permute(1,0) # B,C
        i_vec = self.embedding_i(i_word).unsqueeze(1) # B,1,D
        o_vec = self.embedding_os(o_words) # B, C-1, D
        o_vec_n = self.embedding_os(n_words).neg() # B, N, D
        
        loss_pos = torch.bmm(o_vec, i_vec.permute(0,2,1)).squeeze().sigmoid().log().mean(-1)#.sum(-1) # context sum
        loss_neg = torch.bmm(o_vec_n, i_vec.permute(0,2,1)).squeeze().sigmoid().log().view(-1, context_size, self.neg_n).mean(-1)#.sum(-1)
        
        loss = torch.sum(loss_pos) + torch.sum(loss_neg)  # batch sum
        return -loss  # optimizer minimize loss

if __name__ == '__main__':
    # Set Logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)
    
    # Seed
    SEED = 42 # predictable random variables
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Test Model


