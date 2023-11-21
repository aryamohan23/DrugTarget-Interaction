import argparse
import os
import pickle
import random
import sys
import time
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
import arguments
import models
from preprocess import get_dataset_dataloader

def read_data(filename, key_dir, train):

    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        labels = {l[0]: float(l[1]) for l in lines}
    with open(f"{key_dir}/test_keys.pkl", "rb") as f:
        test_keys = pickle.load(f)
    if train:
        with open(f"{key_dir}/train_keys.pkl", "rb") as f:
            train_keys = pickle.load(f)
        return train_keys, test_keys, labels
    else:
        return test_keys, labels

def calc_loss(iters, loss_fn):
    sample = next(iters, None)
    keys, affinity = sample["key"], sample["affinity"]
    energies_mat, pred, loss_der1, loss_der2 = model(sample)
    loss = loss_fn(pred.sum(-1), affinity)
    return loss, loss_der1, loss_der2, keys, affinity, pred, energies_mat, sample

def write_result(filename, pred, true, positions_ligand, positions_target, energies_mat):
    with open(filename, "w") as w:
        for k in pred.keys():
            w.write(f"{k}\t{true[k]:.3f}\t")
            w.write(f"{pred[k].sum():.3f}\t")
            w.write(f"{0.0}\t")
            for j in range(pred[k].shape[0]):
                w.write(f"{pred[k][j]:.3f}\t")
            w.write(f"{positions_ligand[k]}\t")
            w.write(f"{positions_target[k]}\t")
            w.write(f"{energies_mat[k]}\t")
            w.write("\n")
            
    return


def process_data(keys, pred, affinity, sample):
    pred, true, pos_ligand, pos_target, energies_mat = {}, {}, {}, {}, {}
    if len(keys) > 0:
        pred = pred.data.cpu().numpy()
        for key, p, a in zip(keys, pred, affinity):
            pred[key] = p
            true[key] = a
            ligand[key] = sample["ligand_pos"]
            pos_target[key] = sample["target_pos"]
            energies_mat[key] = energies_mat
    return pred, true, pos_ligand, pos_target, energies_mat


def run(model, scoring_iter, docking_iter, random_iter, cross_iter):
    loss_scoring_list, loss_der1_list, loss_der2_list, loss_docking_list, loss_aug_list = [],[],[],[],[],

    i_batch = 0
    while True:
        print('Batch: ', i_batch)
        model.zero_grad()
        total_loss = 0.0


        ## SCORING
        loss_scoring, loss_der1, loss_der2, keys_scoring, affinity_scoring, pred_scoring, energies_mat_scoring, sample_scoring = calc_loss(scoring_iter, loss_fn)
        total_loss += loss
        total_loss += loss_der1.sum() 
        total_loss += loss_der2.sum() 

        ## DOCKING
        loss_docking, _, _, keys_docking, affinity_docking, pred_docking, energies_mat_docking, sample_docking = calc_loss(docking_iter, loss_fn)
        total_loss += loss_docking * 10 # hyper parameter tune

        ## RANDOM
        loss_random, _, _, keys_random, affinity_random, pred_random, energies_mat_random, sample_random = calc_loss(random_iter, loss_fn)
        total_loss += loss_random * 5 # hyper parameter tune

        ## CROSS
        loss_cross, _, _ , keys_cross, affinity_cross, pred_cross, energies_mat_cross, sample_cross = calc_loss(cross_iter, loss_fn)
        total_loss += loss_cross * 5 # hyper parameter tune

        if is_training:
            total_loss.backward()
            optimizer.step()

        loss_scoring_list.append(loss_scoring)
        loss_der1_list.append(loss_der1)
        loss_der2_list.append(loss_der2)
        loss_docking_list.append(loss_docking)
        loss_aug_list.append(loss_random)
        loss_aug_list.append(loss_cross)

        pred_scoring, true_scoring, pos_ligand_scoring, pos_target_scoring, energies_mat_scoring = process_data(keys_scoring, pred_scoring, affinity_scoring, sample_scoring)
        pred_docking, true_docking, pos_ligand_docking, pos_target_docking, energies_mat_docking = process_data(keys_docking, pred_docking, affinity_docking, sample_docking)
        pred_random, true_random, pos_ligand_random, pos_target_random, energies_mat_random = process_data(keys_random, pred_random, affinity_random, sample_random)
        pred_cross, true_cross, pos_ligand_cross, pos_target_cross, energies_mat_cross = process_data(keys_cross, pred_cross, affinity_cross, sample_cross)

        i_batch += 1
    
    return (
        np.mean(loss_scoring_list), np.mean(loss_der1_list) , np.mean(loss_der2_list) , np.mean(loss_docking_list), np.mean(loss_aug_list), 
        pred_scoring, true_scoring, pos_ligand_scoring, pos_target_scoring, energies_mat_scoring,
        pred_docking, true_docking, pos_ligand_docking, pos_target_docking, energies_mat_docking,
        pred_random, true_random, pos_ligand_random, pos_target_random, energies_mat_random,
        pred_cross, true_cross, pos_ligand_cross, pos_target_cross, energies_mat_cross
    
    )

# Read labels
train_keys_scoring, test_keys_scoring, labels_scoring = read_data('affinity_scoring.txt', 'scoring/keys/')
train_keys_docking, test_keys_docking, labels_docking = read_data('affinity_docking.txt', 'docking/keys/')
train_keys_random, test_keys_random, labels_random = read_data('affinity_random.txt', 'random/keys/')
train_keys_cross, test_keys_cross, labels_cross = read_data('affinity_cross.txt', 'cross/keys/')

# Model
model = models.Model()

# Dataloader
train_dataset_scoring, train_dataloader_scoring = get_dataset_dataloader(train_keys_scoring, 'scoring/data/', labels_scoring, 32)
train_dataset_docking, train_dataloader_docking = get_dataset_dataloader(train_keys_docking, 'docking/data/', labels_docking, 32)
train_dataset_random, train_dataloader_random = get_dataset_dataloader(train_keys_random, 'random/data/', labels_random, 32)
train_dataset_cross, train_dataloader_cross = get_dataset_dataloader(train_keys_cross, 'cross/data/', labels_cross, 32)
# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# train
for epoch in range(0, 1000):
    print('Epoch:', epoch)
    st = time.time()
    # iterator
    train_data_iter_scoring, train_data_iter_docking, train_data_iter_random, train_data_iter_cross = (iter(train_dataloader_scoring), iter(train_dataloader_docking), iter(train_dataloader_random), iter(train_dataloader_cross),)

    # Train
    train_losses, train_losses_der1, train_losses_der2, train_losses_docking, train_losses_screening,
    pred_scoring, true_scoring, pos_ligand_scoring, pos_target_scoring, energies_mat_scoring,
    pred_docking, true_docking, pos_ligand_docking, pos_target_docking, energies_mat_docking,
    pred_random, true_random, pos_ligand_random, pos_target_random, energies_mat_random,
    pred_cross, true_cross, pos_ligand_cross, pos_target_cross, energies_mat_cross = 

    run (model, train_data_iter_scoring, train_data_iter_docking, train_data_iter_random, train_data_iter_cross)

    # Write prediction
    write_result('result/scoring_train.txt', pred_scoring, true_scoring, pos_ligand_scoring, pos_target_scoring, energies_mat_scoring)
    write_result('result/docking_train.txt', pred_docking, true_docking, pos_ligand_docking, pos_target_docking, energies_mat_docking)
    write_result('result/random_train.txt', pred_random, true_random, pos_ligand_random, pos_target_random, energies_mat_random)
    write_result('result/cross_train.txt', pred_cross, true_cross, pos_ligand_cross, pos_target_cross, energies_mat_cross)

    end = time.time()

    # Cal R2
    train_r2_scoring = r2_score(list(true_scoring.values()), list(pred_scoring.values()))
    train_r2_docking = r2_score(list(true_docking.values()), list(pred_docking.values()))
    train_r2_random = r2_score(list(true_random.values()), list(pred_random.values()))
    train_r2_cross = r2_score(list(true_cross.values()), list(pred_cross.values()))


    print('\nLoss:', train_losses, train_losses_der1, train_losses_der2, train_losses_docking, train_losses_screening)

    end = time.time()