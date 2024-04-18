import os 
import numpy as np

def get_pairs(data, SEED): # data = ['path/to/data/img_0_000001.jpg', 'path/to/data/img_0_000002.jpg', ...] for all S{x}/Seq{y}.
    np.random.seed(SEED)
    pairs = np.random.choice(data, size=(12234, 2), replace=False) # choose the same number of pairs as in the original paper
    return pairs

# S = [f'S{i}' for i in range(1,3)]
# Seq = [f'Seq{i}' for i in range(1,3)]
# base_folder = './data/mpi_inf_3dhp/subset/'
def get_data(S, Seq, base_folder):
    data = np.array([])
    for s in S:
        for seq in Seq:
            data = data.concatenate(data, np.array(os.listdir(f'{base_folder}{s}/{seq}/imageSequence/frames_0/')))
    return np.array(data)

if __name__=='__main__':
    S = [f'S{i}' for i in range(1,3)]
    Seq = [f'Seq{i}' for i in range(1,3)]
    base_folder = './data/mpi_inf_3dhp/subset/'
    data = get_data(S, Seq, base_folder)
    print(data.shape)
