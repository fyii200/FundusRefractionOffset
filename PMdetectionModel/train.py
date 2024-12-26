#!/usr/bin/env python
"""
This is an executable script for cross-valiating and training the model for pathologic myopia detection.

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib; import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import BCEWithLogitsLoss
from utils import augmentation, trainStep, valStep
from model import ResNet18
from dataset import fundusDataset
from collections import defaultdict
from tqdm import trange

join   = os.path.join
device = "cuda" if torch.cuda.is_available() else "mps"
root   = os.path.abspath(join(os.getcwd(), "..", ".."))   # Project parent directory
matplotlib.use("Agg")                                     # Turn off plot display

########################################## SETTINGS ###########################################
parser = argparse.ArgumentParser(description = "Pathologic myopia detector")
parser.add_argument("--dataFile", 
                    help    = "Full path to the csv file containing names of input images and their groundtruth labels",
                    type    = str, 
                    default = join(root, "data", "gradedPMfull.csv")) 
parser.add_argument("--imageDir", 
                    help    = "path to folder containing training images",
                    type    = str, 
                    default = os.path.abspath(join(root, "..", "UKB_PM", "images", "full"))) 
parser.add_argument("--weightDir", 
                    help    = "Path to folder in which trained weights are to be saved",
                    type    = str, 
                    default = "weights")
parser.add_argument("--customWeightsPath", 
                    help    = "Full path to the checkpoint file containing weights from the model trained on MMAC dataset",
                    type    = str, 
                    default = join("weights", "pretrainedWeights.pth"))
parser.add_argument("--croppedSize", 
                    help    = "Desired size of the centre crop for fundus image to remove the surrounding black border; default is 1400 by 1400 pixels",
                    type    = int, 
                    default = 1400)
parser.add_argument("--resizedDim", 
                    help    = "Desired input size of fundus images; default is 256 by 256 pixels",
                    type    = int, 
                    default = 256)
parser.add_argument("--numClasses", 
                    help    = "Number of prediction classes; default is binary (2)",
                    type    = int, 
                    default = 2)
parser.add_argument("--batchSize", 
                    help    = "batch size; default is 16",
                    type    = int, 
                    default = 16)
parser.add_argument("--numWorkers", 
                    help    = "number of workers for dataloader; default is 4",
                    type    = int, 
                    default = 4)
parser.add_argument("--lr", 
                    help    = "Adam optimiser's initial learning rate; default is 5e-4",
                    type    = float, 
                    default = 5e-4)
parser.add_argument("--weightDecay", 
                    help    = "Adam optimiser's weight decay; default is 5e-4",
                    type    = float, 
                    default = 5e-4)
parser.add_argument("--betas1", 
                    help    = "Adam optimiser's 1st beta coefficient; default is 0.9",
                    type    = int, 
                    default = 0.9)
parser.add_argument("--betas2", 
                    help    = "Adam optimiser's 2nd beta coefficient; default is 0.999",
                    type    = int, 
                    default = 0.999)
parser.add_argument("--eps", 
                    help    = "Adam optimiser's epsilon (for numerical stability); default is 1e-8",
                    type    = float, 
                    default = 1e-8)
parser.add_argument("--numEpochs", 
                    help    = "total number of epochs; default is 20",
                    type    = int, 
                    default = 20)
parser.add_argument("--kFolds", 
                    help    = "Number of k-folds; default is 5",
                    type    = int, 
                    default = 5)
parser.add_argument("--mixupProb", 
                    help    = "probability of applying mixup augmentation; default is 0",
                    type    = float, 
                    default = 0)
parser.add_argument("--mixupAlpha", 
                    help    = "larger values result in greater mix-up between images; default is 0.4",
                    type    = float, 
                    default = 0.4)
parser.add_argument("--useRAP", 
                    help = "Set to True if using UK Biobank's Research Analysis Platform (False if running locally)", 
                    action="store_true")   # False if not called
parser.add_argument("--TTA",
                    help   = "Apply test-time augmentation at inference time",
                    action = "store_true") # False if not called
parser.add_argument("--plotLosses",
                    help   = "plot and save train + validation loss vs epoch; default is True",
                    action = "store_true") # False if not called

args = parser.parse_args()

################################################## SETUP ###################################################
# Read dataframe with groundtruth labels for pathologic myopia
d = pd.read_csv(join(args.dataFile), low_memory = False)
d = d[~pd.isna(d.PMbinary_eyeLevel) & (d.PMbinary_eyeLevel != "Reject")]
# Make sure groundtruth labels are integers rather than strings
d["PMbinary_eyeLevel"] = d["PMbinary_eyeLevel"].replace({"TRUE": 1, "FALSE": 0})
y                      = np.array(d["PMbinary_eyeLevel"]) 

# Initialise dataset
fullDataset = fundusDataset(imageDir    = args.imageDir,
                            useRAP      = args.useRAP,
                            dataFrame   = d,
                            inference   = False,
                            croppedSize = (args.croppedSize,args.croppedSize), 
                            resizedDim  = (args.resizedDim,args.resizedDim), 
                            numClasses  = args.numClasses)

# # Initialise a cross entropy loss function
# posWeight = torch.tensor(sum(y == 0) / sum(y == 1))
# lossFn    = BCEWithLogitsLoss(reduction = "mean", pos_weight = posWeight)
lossFn    = BCEWithLogitsLoss(reduction = "mean")

# Initialise function for regular data augmentations (random rotation, horizontal flip & brightness/saturation jitter)
augmentFn = augmentation(angles = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30])

# Create an empty default dictionary to store training metrics
metrics   = defaultdict(list)

# Define the K-fold stratified cross validator
kfold     = StratifiedKFold(n_splits = args.kFolds, shuffle = True, random_state = 10)

# Create figure handle
if args.plotLosses:
    fig, ax = plt.subplots(args.kFolds, figsize=(10, 20))

####################################### K-FOLD CROSS VALIDATION BEGINS #######################################
for fold, (trainIDs, valIDs) in enumerate(kfold.split(np.zeros(len(y)), y)):
    
    print("\n--------------"), print(f"Fold {fold + 1}"), print("--------------")
    
    # Initialise train and validation dataloaders
    trainDataloader = DataLoader(fullDataset, 
                                 batch_size   = args.batchSize, 
                                 num_workers  = args.numWorkers,
                                 sampler      = SubsetRandomSampler(trainIDs))
    valDataloader   = DataLoader(fullDataset, 
                                 batch_size   = args.batchSize, 
                                 num_workers  = args.numWorkers,
                                 sampler      = SubsetRandomSampler(valIDs))
    
    # Initialise a pretrained ResNet-18 model
    model = ResNet18(args.customWeightsPath,
                     imageNet   = True,
                     numClasses = args.numClasses)
    model.to(device)
    
    # Initialise Adam optimiser with tuned lr, weight_decay, betas & eps values
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr           = args.lr,
                                 weight_decay = args.weightDecay,
                                 betas        = (args.betas1, args.betas2),
                                 eps          = args.eps)
    
    # Initialise cosine annealing scheduler
    LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.numEpochs)
    
    # Progress bar
    epochIter = trange(0, args.numEpochs, desc="Total number of epochs per fold") 
    for i in epochIter:
        metrics["fold"].append(fold+1)
        metrics["epoch"].append(i+1)
        ##### TRAINING STEP #####
        trainEpochLoss, trainEpochAUC = trainStep(trainDataloader, model, args.mixupProb, args.mixupAlpha, augmentFn, lossFn, optimiser, LRscheduler, device)
        metrics["trainEpochLoss"].append(trainEpochLoss)
        metrics["trainEpochAUC"].append(trainEpochAUC)

        ##### VALIDATION STEP #####
        valEpochLoss, sensitivities, specificities, valEpochAUC = valStep(valDataloader, model, lossFn, args.TTA, device)
        [metrics[name].append(value) for name, value in sensitivities.items()]
        [metrics[name].append(value) for name, value in specificities.items()]                
        metrics["valEpochLoss"].append(valEpochLoss)
        metrics["valEpochAUC"].append(valEpochAUC)  
        
        # Print and save metrics
        print("Train loss: " + str(round(trainEpochLoss, 3))                        + " | " + 
              "Train AUC: "  + str(round(trainEpochAUC*100, 1))                     + " | " + 
              "Val loss: "   + str(round(valEpochLoss, 3))                          + " | " + 
              "Val sens: "   + str(round(sensitivities["valEpochSens0.20"]*100, 1)) + " | " + 
              "Val spec: "   + str(round(specificities["valEpochSpec0.20"]*100, 1)) 
             )
        metricsDf = pd.DataFrame(metrics)
        metricsDf.to_csv(join("metrics", "PMdetectionCrossVal.csv"), index = False)

        # Plot loss vs epoch
        if args.plotLosses:
            ax[fold].plot(metricsDf[metricsDf.fold == fold + 1].trainEpochLoss, label = "train", color = "r") 
            ax[fold].plot(metricsDf[metricsDf.fold == fold + 1].valEpochLoss, label = "validation", color = "b") 
            ax[fold].set_title("Fold " + str(fold+1))
            # Remove surrounding frame
            ax[fold].spines["left"].set_visible(False), ax[fold].spines["right"].set_visible(False), ax[fold].spines["top"].set_visible(False), ax[fold].spines["bottom"].set_visible(False)
            ax[fold].legend(loc="upper right") if i == 0 else None
            plt.savefig(join("metrics", "epochLosses.png"), bbox_inches = "tight", pad_inches = 0, dpi = 200)

plt.close(fig) if args.plotLosses else None            

# Determine which epoch in "metricsDf" yielded the lowest mean validation loss across the k folds        
meanValLossPerEpoch = metricsDf.groupby("epoch")["valEpochLoss"].mean()
bestEpoch           = meanValLossPerEpoch.idxmin() # 'bestEpoch' is NOT zero-indexed (i.e. first epoch is 1, not 0)
print(f"\nEpoch {bestEpoch} yielded the lowest mean validation loss across {args.kFolds} folds, with a value of {meanValLossPerEpoch.min():.4f}")        
print("\n------------------------------------------------------------")
print(f"Train on the full dataset for {args.numEpochs} epochs but stop at epoch {bestEpoch}")
print("------------------------------------------------------------")        

############################################# TRAINING begins #############################################
# Initialise dataloader for the full dataset, pretrained ResNet-18 model, Adam optimiser and cosine annealing learning rate scheduler
trainDataloader = DataLoader(fullDataset, 
                             batch_size         = args.batchSize, 
                             num_workers        = args.numWorkers)
model           = ResNet18(args.customWeightsPath, 
                           imageNet             = True, 
                           numClasses           = args.numClasses)

model           = model.to(device)

optimiser       = torch.optim.Adam(model.parameters(), 
                                   lr           = args.lr, 
                                   weight_decay = args.weightDecay, 
                                   betas        = (args.betas1, args.betas2), 
                                   eps          = args.eps)

LRscheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, 
                                                             args.numEpochs)

# Start training: stop and save weights from the "bestEpoch" epoch
for i in trange(0, bestEpoch, desc = "Total number of training epochs") :
    
    trainEpochLoss, trainEpochAUC = trainStep(trainDataloader, model, args.mixupProb, args.mixupAlpha, augmentFn, lossFn, optimiser, LRscheduler, device)
    
    print("Train loss: " + str(round(trainEpochLoss, 3))      + 
          " | Train AUC: " + str(round(trainEpochAUC*100, 1))
         ) 

torch.save(model.state_dict(), join(args.weightDir, "bestEpochWeights.pth"))    

       



