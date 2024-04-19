import os 
import numpy as np
from glob import glob
import torch
import re
def get_pairs(data, SEED): # data = ['path/to/data/img_0_000001.jpg', 'path/to/data/img_0_000002.jpg', ...] for all S{x}/Seq{y}.
    
    return pairs

# S = [f'S{i}' for i in range(1,3)]
# Seq = [f'Seq{i}' for i in range(1,3)]
# base_folder = './data/mpi_inf_3dhp/subset/'
def get_data(S, Seq, base_folder):
    data = np.array([])
    for s in S:
        for seq in Seq:
            data = np.concatenate((data, np.array(os.listdir(f'{base_folder}{s}/{seq}/imageSequence/frames_0/'))))
    return np.array(data)

if __name__=='__main__':
    S = [i for i in range(1,3)]
    Seq = [i for i in range(1,3)]
    path = './data/mpi_inf_3dhp/subset'
    annot=np.array([[ torch.load(f'{path}/S{s}/Seq{seq}/annot.pt') for seq in Seq] for s in S])
    print(annot[0,0])
    # print(idx.shape)
    # idx -= np.array([0,0,1])
    # print(idx[:5])
    # np.random.seed(0)
    # pairs = np.random.choice(data, size=(12234, 2), replace=False) # choose the same number of pairs as in the original paper
    # print(data.shape)
    # pairs = get_pairs(data, 0)
    # print(pairs.shape)
