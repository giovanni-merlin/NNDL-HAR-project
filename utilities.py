"""
    This script contains utilities for the dataset loading and network definition and training
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#TODO
#ci sono vari argomenti inutilizzati che non si sa a cosa servano

class DopplerTransformations:
    def __init__(self, n_views=2):
        self.n_views = n_views

    def time_warp(self, spec, max_warp=5):
        # Time warping simile a quanto visto
        num_frames = spec.size(-1)
        warp = np.random.randint(-max_warp, max_warp)
        if warp != 0:
            spec = torch.cat([spec[:, :, warp:], spec[:, :, :warp]], dim=-1)
        return spec

    def freq_mask(self, spec, freq_mask_param=10):
        return torchaudio.transforms.FrequencyMasking(freq_mask_param)(spec)

    def time_mask(self, spec, time_mask_param=10):
        return torchaudio.transforms.TimeMasking(time_mask_param)(spec)

    def amplitude_scaling(self, spec, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(*scale_range)
        return spec * scale

    def __call__(self, spec):
        specs = []
        for _ in range(self.n_views):
            augmented_spec = spec.clone()
            augmented_spec = self.time_warp(augmented_spec)
            augmented_spec = self.freq_mask(augmented_spec)
            augmented_spec = self.time_mask(augmented_spec)
            augmented_spec = self.amplitude_scaling(augmented_spec)
            specs.append(augmented_spec)
        return specs


class CSIDataset(Dataset):
    def __init__(self, csi_matrix_files, labels_stride, stream_ant, input_shape, transform=None):
        self.csi_matrix_files = csi_matrix_files
        self.labels_stride = labels_stride
        self.stream_ant = stream_ant
        self.input_shape = input_shape
        self.transform = transform
    
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

        # Applica la trasformazione se è definita
        if self.transform:
            csi_data = self.transform(csi_data)

        label_tensor = torch.Tensor([label]).long()
        
        return (csi_data, label_tensor)
    

def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, transform=None, prefetch=True, repeat=False):
    dataset = CSIDataset(csi_matrix_files, labels_stride, stream_ant, input_shape, transform=transform)

    if repeat: # this is not even used!
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

# changed:
# no more cache files (unused), no input_shape, changed output_shape, changed activities

def create_training_set(dir_init, subdirs_init, feature_length_init = 100, sample_length_init = 340, channels_init = 1, batch_size_init = 32, num_tot_init = 4, activities_init = 'E,L,W,R,J'):
    """
    Create the training set from the raw data.
    We can specify the input dimension, the batch size, and other things.
    Returns a Dataloader object with the input and label batches to iterate on.
    Inputs: dir_init: path of the data directory (str)
            subdirs_init: path of the data subdirectories (str)
            feature_length_init: length along the feature dimension (height)
            sample_length_init: length along the time dimension (width)
            channels_init: number of channels
            batch_size_init: number of samples in a batch
            num_tot_init: number of antenna * number of spatial streams
            activities_init: activities to be considered
    Return: Dataloader obj
    """

    activities = activities_init 
    subdirs_training = subdirs_init
    labels_train = []
    all_files_train = []
    sample_length = sample_length_init
    feature_length = feature_length_init
    channels = channels_init
    num_antennas = num_tot_init
    input_network = (sample_length, feature_length, channels) # maybe better call input shape this?
    batch_size = batch_size_init
    output_shape = len(activities.split(','))
    labels_considered = np.arange(output_shape)

    suffix = '.txt'

    for sdir in subdirs_training.split(','):
        
        #dir_train = dir_init + sdir + '/train_antennas_' + str(activities) + '/'
        name_labels = dir_init + sdir + '/labels_train_antennas_' + str(activities) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_train.extend(pickle.load(fp))
        name_f = dir_init + sdir + '/files_train_antennas_' + str(activities) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_train.extend(pickle.load(fp))

    # create the train dataset
    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                            labels_considered]
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                                labels_considered]

    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas)

    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                stream_ant_train, input_network, batch_size,
                                                shuffle=True)
    
    return dataset_csi_train


def create_validation_set(dir_init, subdirs_init, feature_length_init = 100, sample_length_init = 340, channels_init = 1, batch_size_init = 32, num_tot_init = 4, activities_init = 'E,L,W,R,J'):
    """
    Create the validation set from the raw data.
    We can specify the input dimension, the batch size, and other things.
    Returns a Dataloader object with the input and label batches to iterate on.
    Inputs: dir_init: path of the data directory (str)
            subdirs_init: path of the data subdirectories (str)
            feature_length_init: length along the feature dimension (height)
            sample_length_init: length along the time dimension (width)
            channels_init: number of channels
            batch_size_init: number of samples in a batch
            num_tot_init: number of antenna * number of spatial streams
            activities_init: activities to be considered
    Return: Dataloader obj
    """

    activities = activities_init 
    subdirs_training = subdirs_init
    labels_val = []
    all_files_val = []
    sample_length = sample_length_init
    feature_length = feature_length_init
    channels = channels_init
    num_antennas = num_tot_init
    input_network = (sample_length, feature_length, channels) # maybe better call input shape this?
    batch_size = batch_size_init
    output_shape = len(activities.split(','))
    labels_considered = np.arange(output_shape)

    suffix = '.txt'

    for sdir in subdirs_training.split(','):

        #dir_val = dir_init + sdir + '/val_antennas_' + str(activities) + '/'
        name_labels = dir_init + sdir + '/labels_val_antennas_' + str(activities) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_val.extend(pickle.load(fp))
        name_f = dir_init + sdir + '/files_val_antennas_' + str(activities) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_val.extend(pickle.load(fp))

    # create the validation dataset
    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                            labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                            labels_considered]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)

    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                            stream_ant_val, input_network, batch_size,
                                            shuffle=False)
    return dataset_csi_val


def create_test_set(dir_init, subdirs_init, feature_length_init = 100, sample_length_init = 340, channels_init = 1, batch_size_init = 32, num_tot_init = 4, activities_init = 'E,L,W,R,J'):
    """
    Create the test set from the raw data.
    We can specify the input dimension, the batch size, and other things.
    Returns a Dataloader object with the input and label batches to iterate on.
    Inputs: dir_init: path of the data directory (str)
            subdirs_init: path of the data subdirectories (str)
            feature_length_init: length along the feature dimension (height)
            sample_length_init: length along the time dimension (width)
            channels_init: number of channels
            batch_size_init: number of samples in a batch
            num_tot_init: number of antenna * number of spatial streams
            activities_init: activities to be considered
    Return: Dataloader obj
    """

    activities = activities_init 
    subdirs_training = subdirs_init
    labels_test = []
    all_files_test = []
    sample_length = sample_length_init
    feature_length = feature_length_init
    channels = channels_init
    num_antennas = num_tot_init
    input_network = (sample_length, feature_length, channels) # maybe better call input shape this?
    batch_size = batch_size_init
    output_shape = len(activities.split(','))
    labels_considered = np.arange(output_shape)

    suffix = '.txt'

    for sdir in subdirs_training.split(','):
        #dir_test = dir_init + sdir + '/test_antennas_' + str(activities) + '/'
        name_labels = dir_init + sdir + '/labels_test_antennas_' + str(activities) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_test.extend(pickle.load(fp))
        name_f = dir_init + sdir + '/files_test_antennas_' + str(activities) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_test.extend(pickle.load(fp))

    # create the test dataset
    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)

    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                                stream_ant_test, input_network, batch_size,
                                                shuffle=False)
    return dataset_csi_test