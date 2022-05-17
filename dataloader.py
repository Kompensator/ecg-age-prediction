import math
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import os
from tqdm import tqdm
import csv
from multiprocessing import Pool
import random


def load_all_h5py():
    path = r'C:\Users\dingyi.zhang\Downloads\CODE-15%'
    files = os.listdir(path)[1:]
    numbers = []
    for f in files:
        numbers.append(int(f.split('_')[1].split('.')[0][4:]))
    c = list(zip(files, numbers))
    c = sorted(c, key=lambda x: x[1])
    files, numbers = zip(*c)

    print("Loading all h5py datasets...")
    started = False
    for f in tqdm(files):
        if '.hdf5' in f:
            fpath = os.path.join(path, f)
            with h5py.File(fpath, mode='r') as f:
                if not started:
                    all_data = np.array(f['tracings'])
                    all_ID = np.array(f["exam_id"])
                    started = True

                else:
                    data = np.array(f['tracings'])
                    ID = np.array(f['exam_id'])
                    all_data = np.concatenate((all_data, data), axis=0)
                    all_ID = np.concatenate((all_ID, ID), axis=0)
    return all_data, all_ID


def CODE_15():
    all_ECG, all_ID = load_all_h5py()
    all_ID = list(all_ID)
    all_ECG = list(all_ECG)

    c = list(zip(all_ECG, all_ID))
    c = sorted(c, key=lambda x: x[1])
    all_ECG, all_ID = zip(*c)

    data = []
    with open(r'C:\Users\dingyi.zhang\Downloads\CODE-15%\exams.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    data = data[1:]

    ECG = []
    AGE = []

    i = 0
    j = 0
    while i < len(all_ID) and j < len(data):
        if all_ID[i] == int(data[j][0]):
            ECG.append(all_ECG[j])
            AGE.append(int(data[j][1]))
            i += 1
            j += 1
        elif all_ID[i] > int(data[j][0]):
            j += 1
        elif all_ID[i] < int(data[j][0]):
            i += 1

    c = list(zip(ECG, AGE))
    random.seed(42)
    random.shuffle(c)
    ECG, AGE = zip(*c)

    ECG = np.array(ECG)
    AGE = np.array(AGE)
    print("ECG shape {}\t AGE shape {}".format(ECG.shape, AGE.shape))

    return ECG, AGE
    

class BatchDataloader:
    def __init__(self, *tensors, bs=1, mask=None):
        nonzero_idx, = np.nonzero(mask)
        self.tensors = tensors
        self.batch_size = bs
        self.mask = mask
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)+1
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.start = end
        self.sum += sum(batch_mask)
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count


if __name__ == "__main__":
    CODE_15()