import numpy as np
import hp as hp
import analogy as ag

from numpy import linalg as Lig
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import pickle
from collections import OrderedDict
from itertools import chain
import argparse

# Hyper-parameters
input_size = 100
output_size = 100
num_epochs = 100
learning_rate = 0.001


MODEL = {
    # not splited Hyperboloid model trained squ-lorenz
    'poin': './model/100_model_pdist_neg5_6.pkl',
    # 'prod': './model/10x10_klein_model_mixed_neg5_4.pkl',
    # 'prod': './model/10x10_pdist_model_mixed_neg5_1.pkl',
    'prod': './model/10x10_model_mixed_neg5_1.pkl',
    'klein': './model/model_klein100_neg5_17.pkl',
    'hyp': './model/model_ldist1_neg5_1_hp.pkl',  # original
    # 'hyp' : './model/model_lorentz_neg5_4.pkl'
}

ANALOGY_FILE = './analogy_task_file/mixed_questions-words.txt'
path_vocab = './model/vocab.pkl'


def make_time_ax(weights):
    w_t = np.sqrt((weights**2).sum(axis=1)+1.)
    weight = np.hstack((weights, w_t.reshape(-1, 1)))
    return weight


def delete_time_ax(weights):
    weight = np.delete(weights, -1, 1)  # delete last column
    weight = weight / \
        (1. + np.sqrt((weight**2).sum(axis=1, keepdims=True) + 1.))

    return weight


def read_questions(questions_file):
    """ read google questions sheet"""

    questions = []
    with open(questions_file) as f:
        for line in f:
            if line[0] == ':' or line == '':
                continue

            w1, w2, w3, w4 = line.strip('\n').split(' ')

            questions.append([w1, w2, w3, w4])
    return questions


def word2vec(w1, w2, w3, w4, word2idx, weights):
    """ change word into vector"""
    w1, w2, w3, w4 = w1.lower(), w2.lower(), w3.lower(), w4.lower()
    return weights[word2idx[w1]], weights[word2idx[w2]],\
        weights[word2idx[w3]], weights[word2idx[w4]]


def mob_add(u, v, c):
    """moebius add"""
    numerator = (1. + 2. * c * np.dot(u, v) + c * Lig.norm(v)
                 ** 2) * u + (1. - c * Lig.norm(u)**2) * v
    denominator = 1. + 2. * c * \
        np.dot(u, v) + c**2 * Lig.norm(v)**2 * Lig.norm(u)**2
    return numerator / denominator


def mob_gyr(a, b, c, ver=True):
    if ver:
        diff = mob_add(-(a), b, 1.)
        p1 = mob_add(-(a), diff, 1.)
        p2 = mob_add(c, p1, 1.)

        pp1 = mob_add(c, -(a), 1.)
        pp2 = mob_add(-(pp1), p2, 1.)
        result = mob_add(c, pp2, 1.)
        return result

    else:
        diff = mob_add(b, -(a), 1.)
        p1 = mob_add(diff, -(a), 1.)
        p2 = mob_add(c, p1, 1.)
        pp1 = mob_add(c, -(a), 1.)
        pp2 = mob_add(pp1, p2, 1.)
        return pp2


def model_load(path_model, path_vocab):
    #model = torch.load(path_model, map_location='cuda:0')
    #weights = model['embed.weight']
    weights = torch.load(path_model, map_location='cuda:0')
    weights = weights.cpu().numpy().astype(np.float32)
    #del model
    with open(path_vocab, 'rb') as r:
        vocab = pickle.load(r)

    idx2word = {idx: w for idx, w in enumerate(vocab)}
    word2idx = {w: idx for idx, w in enumerate(vocab)}
    return weights, idx2word, word2idx

# test poincare


def test(weights):
    for i in range(weights.shape[1]):
        print(Lig.norm(weights[i]))

# test hyperboloid


def ldot(u, v, keepdims=True):
    uv = u * v
    if len(uv.shape) == 2:
        uv[:, -1] = uv[:, -1]*-1
        return np.sum(uv, axis=1, keepdims=True)

    else:
        uv[-1] = uv[-1]*-1
        return np.sum(uv)


def test_hyp(weights):
    w = 0
    for i in range(weights.shape[1]):
        print(ldot(weights[i], weights[i]))


def get_date():

    questions = read_questions(ANALOGY_FILE)
    # version 1 : changed equation

    list_changed = []
    for q in tqdm(questions):

        try:
            v1, v2, v3, v4 = word2vec(
                q[0], q[1], q[2], q[3], word2idx, weights_poin)

            z_ = mob_gyr(v1, v3, v2)
            list_changed.append(z_)

        except Exception as e:
            continue

        # version 2 : equation (wrong equ)

        list_wrong = []
        for q in tqdm(questions):

        try:
        v1, v2, v3, v4 = word2vec(
            q[0], q[1], q[2], q[3], word2idx, weights_poin)

        z_ = mob_gyr(v1, v3, v2)
        list_wrong.append(z_)

        except Exception as e:
        continue


# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                                   1, num_epochs, loss.item()))
