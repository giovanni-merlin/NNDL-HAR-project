"""
    This script contains utilities for network definition and training
"""

import os
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def create_subfolders_from_ckpt(read_path):
    files = os.listdir(read_path)
    ckpt_files = [f for f in files if f.endswith('.ckpt')]
    
    # Create subfolders for each .ckpt file
    for ckpt_file in ckpt_files:
        subfolder_name = os.path.splitext(ckpt_file)[0]
        subfolder_path = os.path.join(read_path, subfolder_name)
        
        # Create the subfolder if it doesn't exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created subfolder: {subfolder_path}")
        else:
            print(f"Subfolder already exists: {subfolder_path}")

# CONTRASTIVE LEARNING #

def NT_Xent_self(features_batch, temperature, mode='train'):
    """
    Takes in input a features_batch tensor of shape (BatchSize * 2, feature_dim), the temperature parameter,
    and computes the NT_Xent_loss
    """

    # Calculate cosine similarity between all possible couples of examples in the features_batch tensor
    # Result must be a (BatchSize*2, BatchSize*2) tensor

    cos_sim = F.cosine_similarity(features_batch[:,None,:], features_batch[None,:,:], dim=-1) #uses broadcasting, compute similarity along the last dimension
    # cos_sim is a matrix of dimension (2*BatchSize)^2, symmetric with 1s on the diagonal

    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find the positive example, we know that it is batch_size away from the original example
    # matrix of boolean
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

    # NT_Xent loss
    cos_sim = cos_sim / temperature

    # cos_sim[pos_mask] returns an array with only the cosine similarity between the positive examples
    # logsumexp is the same for both dims since cos_sim is symmetric
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1) 
    nll = nll.mean() # average over the batch

    # Get ranking position of positive example
    comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                            cos_sim.masked_fill(pos_mask, -9e15)],
                            dim=-1)
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

    acc_top1 = (sim_argsort == 0).float().mean() # quante volte azzecca la chiave positiva in media
    acc_top5 = (sim_argsort < 5).float().mean() # quante volte azzecca la chiave positiva tra le prime 5 in media

    return nll, acc_top1, acc_top5

def NT_Xent_sup(features_batch, temperature):
    """
    Supervised implementation of the NT-Xent loss for 3 positive keys

    Takes in input a features_batch tensor of shape (BatchSize * 4, feature_dim), the temperature parameter,
    and computes the NT_Xent_loss
    """

    # Calculate cosine similarity between all possible couples of examples in the features_batch tensor
    # Result must be a (BatchSize*4, BatchSize*4) tensor

    cos_sim = F.cosine_similarity(features_batch[:, None, :], features_batch[None, :, :], dim=-1)

    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find the positive examples, we know that they are batch_size away from the original example
    batch_size = cos_sim.shape[0] // 4
    pos_mask = torch.zeros_like(self_mask, device=cos_sim.device)
    for i in range(1, 4):
        pos_mask |= self_mask.roll(shifts=batch_size * i, dims=0) # bitwise OR to accumulate the position of the 3 positive keys

    cos_sim = cos_sim / temperature

    # for every sample sum the cosine similarities of its positive keys --> we have to reshape
    # and multiply the logsumexp by 3 to account for the correct number of positive keys
    sum_positives = cos_sim[pos_mask].reshape(-1,3).sum(dim=1)
    nll = -sum_positives + torch.logsumexp(cos_sim, dim=-1)*3
    nll = nll.mean() / 3

    # Get ranking position of ALL THE THREE POSITIVE EXAMPLES
    comb_sim = torch.cat([cos_sim[pos_mask].reshape(-1,3), cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
    sim_argsort = comb_sim.argsort(dim=-1, descending=True)
    
    # check maybe not need a tensor
    idx0 = torch.tensor([np.where(row == 0)[0][0] for row in sim_argsort])
    idx1 = torch.tensor([np.where(row == 1)[0][0] for row in sim_argsort])
    idx2 = torch.tensor([np.where(row == 2)[0][0] for row in sim_argsort]) # we need the 0,0 because it returns a tuple
    pos_indices = torch.stack((idx0, idx1, idx2), dim=-1).cpu().numpy()

    target_set = {0,1,2}

    # Check each row in the array, compute the accuracy
    acc_top1 = torch.tensor([set(row) == target_set for row in pos_indices]).float().mean()
    
    return nll, acc_top1


def train_contrastive_self(model, device, train_loader, optimizer, lr_scheduler, epoch, loss_temperature):
    train_losses = []
    train_top1_accs = []

    model.train()
    model.to(device)
    for i, batch in enumerate(train_loader):
        imgs, _ = batch

        cat_imgs = torch.cat(imgs, dim=0).to(device)

        # Compute the features
        features = model(cat_imgs)

        # Compute the loss together with the accuracy metrics, and store them in the lists above
        nce_loss, acc_top1, acc_top5 = NT_Xent_self(features, temperature=loss_temperature)
        train_losses.append(nce_loss.item())
        train_top1_accs.append(acc_top1.item())

        # Backpropagate the loss and perform the optimization step
        optimizer.zero_grad()
        nce_loss.backward()
        optimizer.step()

    print(f"Train Epoch: {epoch},  \tLoss: {np.mean(train_losses).item():.6f}, \tTop1_Acc: {np.mean(train_top1_accs).item():.6f}")
    lr_scheduler.step()
    return np.mean(train_losses), np.mean(train_top1_accs)

@torch.no_grad()
def valid_constrastive_self(model, device, val_loader, epoch, loss_temperature):
    model.eval()
    with torch.no_grad():
        val_losses = []
        val_top1_accs = []
        for i, batch in enumerate(val_loader):
            imgs, _ = batch

            # Concatenate the images
            imgs = torch.cat(imgs, dim=0).to(device)

            # Compute the features
            features = model(imgs)

            # Compute loss and accuracies, and store them
            nce_loss, acc_top1, acc_top5 = NT_Xent_self(features, temperature=loss_temperature)
            val_losses.append(nce_loss.item())
            val_top1_accs.append(acc_top1.item())
            
        print(f"Valid Epoch: {epoch},  \tLoss: {np.mean(val_losses).item():.6f}, \tTop1_Acc: {np.mean(val_top1_accs).item():.6f}")   
    return np.mean(val_losses), np.mean(val_top1_accs)


def train_contrastive_sup(model, device, train_loader, optimizer, lr_scheduler, epoch, loss_temperature):
    train_losses = []
    train_top1_accs = []

    model.train()
    model.to(device)
    for i, batch in enumerate(train_loader):
        batch_x, batch_y = batch

        # Here the input is different from the self supervised case: batch_x is not a list of two,
        # rather a unique tensor with four channels. Therefore it has to be reshaped 
        # TO BE CHECKED
        
        batch_x_resh = batch_x.reshape(-1, 1, 340, 100).to(device)

        # Compute the features
        features = model(batch_x_resh)

        # Compute the loss together with the accuracy metrics, and store them in the lists above
        nce_loss, acc_top1 = NT_Xent_sup(features, temperature=loss_temperature)
        train_losses.append(nce_loss.item())
        train_top1_accs.append(acc_top1.item())

        # Backpropagate the loss and perform the optimization step
        optimizer.zero_grad()
        nce_loss.backward()
        optimizer.step()

    print(f"Train Epoch: {epoch},  \tLoss: {np.mean(train_losses).item():.6f}, \tTop1_Acc: {np.mean(train_top1_accs).item():.6f}")
    lr_scheduler.step()
    return np.mean(train_losses), np.mean(train_top1_accs)

@torch.no_grad()
def valid_constrastive_sup(model, device, val_loader, epoch, loss_temperature):
    model.eval()
    with torch.no_grad():
        val_losses = []
        val_top1_accs = []
        for i, batch in enumerate(val_loader):
            batch_x, batch_y = batch

            # Here the input is different from the self supervised case: batch_x is not a list of two,
            # rather a unique tensor with four channels. Therefore it has to be reshaped 
            # TO BE CHECKED
        
            batch_x_resh = batch_x.reshape(-1, 1, 340, 100).to(device)

            # Compute the features
            features = model(batch_x_resh)

            # Compute loss and accuracies, and store them
            nce_loss, acc_top1 = NT_Xent_sup(features, temperature=loss_temperature)
            val_losses.append(nce_loss.item())
            val_top1_accs.append(acc_top1.item())
            
        print(f"Valid Epoch: {epoch},  \tLoss: {np.mean(val_losses).item():.6f}, \tTop1_Acc: {np.mean(val_top1_accs).item():.6f}")   
    return np.mean(val_losses), np.mean(val_top1_accs)

# TEST #

# NB: should put the name as a variable so we can save and load in a consistent way?
def train_projection(model, optimizer, train_feats_data, val_feats_data, batch_size, save_dir:str, save_name:str, device, epochs=100):

    model.to(device)
    train_loader = DataLoader(train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_feats_data, batch_size=batch_size, shuffle=False, drop_last=False)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val = np.inf

    best_test_acc = 0
    for epoch in range(epochs):
        losses = []
        accs = []
        model.train()
        for batch in train_loader:
            feats, labels = batch
            feats = feats.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)

            # Forward pass
            preds = model(feats)
            #preds = t.squeeze(preds, dim=1)
            #print(preds.shape)

            loss = F.cross_entropy(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())

            # Backpropagate the loss and perform the optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(np.mean(losses))
        train_accs.append(np.mean(accs))

        print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]:.6f}, Train Accuracy: {train_accs[-1]:.6f}")

        model.eval()
        with torch.no_grad():
            vlosses = []
            vaccs = []
            model.eval()
            for batch in val_loader:
                feats, labels = batch
                feats = feats.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(dim=1)

                # Forward pass
                preds = model(feats)
                #preds = t.squeeze(preds, dim=1)
                #print(preds.shape)

                loss = F.cross_entropy(preds, labels)
                acc = (preds.argmax(dim=-1) == labels).float().mean()
                vlosses.append(loss.item())
                vaccs.append(acc.item())

            val_losses.append(np.mean(vlosses))
            val_accs.append(np.mean(vaccs))
            print(f"Epoch: {epoch}, Val Loss: {val_losses[-1]:.6f}, Val Accuracy: {val_accs[-1]:.6f}")

        if val_losses[-1] < best_val:
            iteration = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            print("Saved Model")
            best_val = val_losses[-1]

        '''
        EARLY STOPPING
        if epoch%10 == 0:
            ## Test set
            print()
            test_acc = []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    feats, labels = batch
                    feats = feats.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    preds = model(feats)
                    acc = (preds.argmax(dim=-1) == labels).float().mean()
                    test_acc.append(acc.item())

                test_acc = np.mean(test_acc)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc.item()
                else:
                    print("Early stop at epoch ", epoch)
                    break
        '''

    print("iteration with best score: ", iteration)

    return train_losses, train_accs, val_losses, val_accs


def prepare_data_features(model, dataset, device):
    """
    Gives the feature representation of the data in "dataset" by the the encoder "model".
    Used to prepare the data for the projection head training and testing.
    """
    with torch.no_grad():
        # Prepare model
        network = deepcopy(model)
        network.classification_layer = nn.Identity() # MIGHT CHANGE THIS, OR PASS AS AN ARGUMENT!
        network.eval()
        network.to(device)

        # Encode all images
        feats, labels = [], []
        for batch_imgs, batch_labels in tqdm(dataset):
            batch_imgs = batch_imgs.to(device)
            batch_feats = network(batch_imgs)
            feats.append(batch_feats.detach().cpu())
            labels.append(batch_labels)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

    return TensorDataset(feats, labels)


def merge_predictions(labels_true_test, labels_prediction_list_test, labels_pred_test, num_antennas=4):
    """
    Merge the predictions from the 4 antennas according to a majority rule.

    Returns the merged predicted labels, to be used to plot the merged confusion matrix.
    """

    #labels_true_merged = np.array(labels_true_test)
    labels_pred_max_merged = np.zeros_like(labels_true_test)

    # iterate over all the labels of the test set, not expanded
    # each iteration is a different input 
    for i_lab in range(len(labels_pred_max_merged)):

        # per ogni input, somma i punteggi delle 4 antenne e trova la predizione migliore
        pred_antennas = labels_prediction_list_test[i_lab * num_antennas:(i_lab + 1) * num_antennas, :].numpy()
        labels_pred_test_merged = np.argmax(np.sum(pred_antennas, axis=0))

        # per ogni input, conta le ricorrenza delle predizioni
        pred_max_antennas = labels_pred_test[i_lab * num_antennas:(i_lab + 1) * num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)

        # se ho un solo label per le 4 antenne -> è il label predetto
        # se ho due labels (non ex aequo) -> scegli quello con più voti
        # se ho più di due labels o due labels ex aequo  -> scegli quello con punteggio più alto
        lab_max_merged = -1
        if len(lab_unique) > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]

            if count_sort[0] == count_sort[1] or len(lab_unique) > 2:
                lab_max_merged = labels_pred_test_merged
            else:
                lab_max_merged = lab_unique_sort[0]
        else:
            lab_max_merged = lab_unique[0] 
        labels_pred_max_merged[i_lab] = lab_max_merged
    
    return labels_pred_max_merged


def plt_confusion_matrix(number_activities, confusion_matrix, lables, title, save_dir=None, save_name=None, PI=None):

    confusion_matrix_normaliz_col = np.transpose(confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(-1, 1)) #NB: prima si chiamava row, ma la normalizzazione è lungo le colonne
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5.5, 4)
    ax = fig.add_axes((0.18, 0.15, 0.6, 0.8))
    im1 = ax.pcolor(np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    confusion_matrix_normaliz_col, cmap='Blues', edgecolors='black', vmin=0, vmax=1)
    if PI:
        ax.set_xlabel('True person', fontsize=18)
        ax.set_ylabel('Predicted person', fontsize=18)
    else:
        ax.set_xlabel('True activity', fontsize=18)
        ax.set_ylabel('Predicted activity', fontsize=18)
    ax.set_xticks(np.linspace(1, number_activities, number_activities))
    ax.set_xticklabels(labels=lables, fontsize=18)
    ax.set_yticks(np.linspace(1, number_activities, number_activities))
    ax.set_yticklabels(labels=lables, fontsize=18, rotation=45)
    ax.set_title(title)

    for x_ax in range(confusion_matrix_normaliz_col.shape[0]):
        for y_ax in range(confusion_matrix_normaliz_col.shape[1]):
            col = 'k'
            value_c = round(confusion_matrix_normaliz_col[x_ax, y_ax], 2)
            if value_c > 0.6:
                col = 'w'
            if value_c > 0:
                ax.text(y_ax + 1, x_ax + 1, '%.2f' % value_c, horizontalalignment='center',
                        verticalalignment='center', fontsize=16, color=col)

    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.ax.set_ylabel('Accuracy', fontsize=18)
    cbar.ax.tick_params(axis="y", labelsize=16)

    #plt.tight_layout()
    if save_dir is not None:
        name_fig = os.path.join(save_dir, 'cm_' + save_name + '.png')
        
        # Check if the file already exists and rename it if necessary
        base_name, extension = os.path.splitext(name_fig)
        counter = 1
        while os.path.exists(name_fig):
            name_fig = f"{base_name}_{counter}{extension}"
            counter += 1
        plt.savefig(name_fig)