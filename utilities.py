"""
    This script contains utilities for the dataset loading and network definition and training
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#TODO
#ci sono vari argomenti inutilizzati che non si sa a cosa servano


class CSIDataset(Dataset):
    def __init__(self, csi_matrix_files, labels_stride, stream_ant, input_shape):
        self.csi_matrix_files = csi_matrix_files
        self.labels_stride = labels_stride
        self.stream_ant = stream_ant
        self.input_shape = input_shape
    
    def __len__(self):
        return len(self.csi_matrix_files)
    
    def __getitem__(self, idx):
        csi_file = self.csi_matrix_files[idx]
        label = self.labels_stride[idx]
        stream = self.stream_ant[idx]
        csi_data = load_data_single(csi_file, stream)
        
        # Per assicurarsi che il tensore abbia la forma corretta
        csi_data = csi_data.view(self.input_shape)
        csi_data = csi_data.permute(2, 0, 1)

        label_tensor = torch.Tensor([label]).float()
        
        return (csi_data, label_tensor)
    

def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, #NB cache file is not used!
                          cache_file=None, prefetch=True, repeat=False):
    dataset = CSIDataset(csi_matrix_files, labels_stride, stream_ant, input_shape)

    if repeat: # this is not even used!
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


def load_data_single(csi_file_t, stream_a):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi_single = matrix_csi[stream_a, ...].T
    if len(matrix_csi_single.shape) < 3:
        matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)

    matrix_csi_single = torch.tensor(matrix_csi_single, dtype=torch.float32) #vedi che fa
    return matrix_csi_single