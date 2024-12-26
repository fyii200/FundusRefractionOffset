#!/usr/bin/env python
"""
This is an executable script for cross-validating and training a deep learning model 
called "fusionet" to predict fundus equivalent refraction from fundus images.

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024
"""

import os
import warnings
import argparse
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from model import fusionet
from dataset import DatasetGenerator
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from utils import trainStep, valStep, weightedMSE, WeightedSubsetRandomSampler
from tqdm import trange
import matplotlib
matplotlib.use("Agg") # turn off plot display
import matplotlib.pyplot as plt

join   = os.path.join
plt    = matplotlib.pyplot
device = "cuda" if torch.cuda.is_available() else "cpu"
root   = os.path.abspath(join(os.getcwd(), "../.."))     # Project parent directory

os.environ["NCCL_DEBUG"]         = "OFF"
os.environ["NCCL_IB_DISABLE"]    = "1"
warnings.filterwarnings("ignore", 
                        category = UserWarning)


#################################### Setting parameters #####################################
parser = argparse.ArgumentParser(description = "Retinal equivalent refraction")
parser.add_argument("--weightsDir", 
                    help    = "directory where model weights are saved",
                    type    = str, 
                    default = "weights") 
parser.add_argument("--imageDir", 
                    help    = "Full path to the directory where input images are saved (set to None if 'useRAP' is True",
                    type    = str, 
                    default = None)
parser.add_argument("--batchSize", 
                    help    = "batch size during training and validation; default is 32",
                    type    = int, 
                    default = 32)
parser.add_argument("--numWorkers", 
                    help    = "number of workers for dataloader; default is 4",
                    type    = int, 
                    default = 4)
parser.add_argument("--lr", 
                    help    = "Adam optimiser's initial learning rate; default is 1e-4",
                    type    = float, 
                    default = 1e-4)
parser.add_argument("--weightDecay", 
                    help    = "Adam optimiser's weight decay; default is 5e-4",
                    type    = float, 
                    default = 5e-4)
parser.add_argument("--fundusCroppedSize", 
                    help    = "Desired size of the centre crop for fundus image to remove the surrounding black border",
                    type    = int, 
                    default = 1400)
parser.add_argument("--resizedDim", 
                    help    = "Desired input size of fundus images and OCT B-scans",
                    type    = int, 
                    default = 512)
parser.add_argument("--kFolds", 
                    help    = "Number of k-folds; default is 5",
                    type    = int, 
                    default = 5)
parser.add_argument("--numEpochs", 
                    help    = "total number of epochs; default is 20",
                    type    = int, 
                    default = 20)
parser.add_argument("--MSEweightFactor", 
                    help    = "Weight factor applied to the weight used in the loss function (larger value = larger penalty)",
                    type    = float, 
                    default = 1.2)
parser.add_argument("--fusion", 
                    help    = "Set to True if fundus images and OCT scans are used as inputs; False if only fundus images are used", 
                    action  = "store_true") # False if not called
parser.add_argument("--useRAP", 
                    help    = "Set to True if using UK Biobank's Research Analysis Platform (False if running locally)", 
                    action  = "store_true") # False if not called
parser.add_argument("--activationInFCL", 
                    help    = "whether tanH activation function should be added to the fully connected layers",
                    action  = "store_true") # False if not called
parser.add_argument("--plotLosses",
                    help    = "plot and save loss vs epoch",
                    action  = "store_true") # False if not called
args = parser.parse_args()


###################################### Train setup #######################################
# Benchmark multiple convolution algorithms and select the fastest
torch.backends.cudnn.benchmark = True    

# Create weights, performance metrics and image name directories if they have not been created
os.makedirs(args.weightsDir) if not os.path.exists(args.weightsDir) else None
os.makedirs("metrics")       if not os.path.exists("metrics")       else None    
os.makedirs("imageNames")    if not os.path.exists("imageNames")    else None    
    
# Read dataframe
data = pd.read_csv(join(root, "data", "retinalRefractionGap_train_new.csv"), low_memory = False)
print(str(len(data)) + " images")

# Get groundtruth (also compute binned groundtruth for stratified cross-validation purposes)
y       = data.SER
yBinned = pd.qcut(y, q = 4, labels = False)   

# Categorise eyes into 3 different SER classes and compute class weights (used to create "trainSampler" below) 
data["SERclass"]                       = [1] * len(data) 
data.loc[data.SER <= -0.5, "SERclass"] = 2
data.loc[data.SER <= -5, "SERclass"]   = 3                   
classProp                              = data["SERclass"].value_counts(normalize = True)          
classWeights                           = [1 / classProp[i] for i in data["SERclass"]]  

print(str(round(classProp.values[0]*100, 1)) + "%" + " Non-myopes | "        + 
      str(round(classProp.values[1]*100, 1)) + "%" + " SER b/w -0.5 & -5 | " +
      str(round(classProp.values[2]*100, 1)) + "%" + " SER ≤ -5" )


# Initialise dataset
fullDataset = DatasetGenerator(dataFrame         = data,
                               imageDir          = args.imageDir,
                               useRAP            = args.useRAP, 
                               train             = True, 
                               getOCT            = args.fusion,                                
                               fundusCroppedSize = (args.fundusCroppedSize, args.fundusCroppedSize), 
                               resizedDim        = (args.resizedDim, args.resizedDim))    
    
# Weighted mean squared error loss function (larger penalty for more myopic SER) 
lossFn      = weightedMSE(args.MSEweightFactor)
    
# Create an empty default dictionary to store training & validation metrics
metrics     = defaultdict(list)    

# Define the K-fold stratified cross validator
kfold       = StratifiedKFold(n_splits = args.kFolds, shuffle = True, random_state = 10)

# Create figure handle
fig, ax     = plt.subplots(args.kFolds, figsize=(10, 20)) if args.plotLosses else (None, None)    
    
    
####################################### K-FOLD CROSS VALIDATION BEGINS #######################################
for fold, (trainIDs, valIDs) in enumerate(kfold.split(np.zeros(len(y)), yBinned)):
    
    print("\n--------------"), print(f"Fold {fold + 1}"), print("--------------")
    
    # Custom data sampler for oversampling eyes with high myopia during training
    trainSampler    = WeightedSubsetRandomSampler(trainIDs, classWeights, replacement = True)
    
    # Initialise train and validation dataloaders
    trainDataLoader = DataLoader(fullDataset, 
                                 batch_size    = args.batchSize, 
                                 num_workers   = args.numWorkers,
                                 sampler       = trainSampler)
    valDataLoader   = DataLoader(fullDataset, 
                                 batch_size    = args.batchSize, 
                                 num_workers   = args.numWorkers,
                                 sampler       = SubsetRandomSampler(valIDs))
    
    # Initialise model
    model           = fusionet(activationInFCL = args.activationInFCL, 
                               fusion          = args.fusion)
    model.to(device)
    
    # parallelise data across multiple GPUs if there's more than 1 GPU
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        
    # Initialise Adam optimiser with tuned lr, weight_decay, betas & eps values
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr           = args.lr,
                                 weight_decay = args.weightDecay)
    
    # Initialise cosine annealing scheduler
    LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = args.numEpochs)
    
    # Progress bar
    epochIter = trange(0, args.numEpochs, desc = "Total number of epochs per fold") 
    for i in epochIter:
        metrics["fold"].append(fold+1)
        metrics["epoch"].append(i+1)

        ##### TRAINING STEP #####
        trainEpochLoss, trainEpochMAE = np.round(trainStep(trainDataLoader, True, args.fusion, model, optimiser, lossFn, LRscheduler, device), 3)
        metrics["trainEpochLoss"].append(trainEpochLoss)
        metrics["trainEpochMAE"].append(trainEpochMAE)      
    
        ##### VALIDATION STEP #####
        valEpochLoss, valEpochMAE = np.round(valStep(valDataLoader, model, args.fusion, lossFn, device), 3)
        metrics["valEpochLoss"].append(valEpochLoss)
        metrics["valEpochMAE"].append(valEpochMAE)      

        # Print and save metrics
        print("\nTrain weighted MSE & MAE: " + 
              str(trainEpochLoss)            + 
              " & "                          + 
              str(trainEpochMAE)             +
              " | Val weighted MSE & MAE : " + 
              str(valEpochLoss)              + 
              " & "                          + 
              str(valEpochMAE) )
        
        metricsDf = pd.DataFrame(metrics)
        metricsDf.to_csv(join("metrics", "CrossVal.csv"), index = False)

        print("^^^^^^^", "Epoch " + str(i+1), " done ^^^^^^^\n")

        # Plot loss vs epoch
        if args.plotLosses:
            ax[fold].plot(metricsDf[metricsDf.fold == fold + 1].trainEpochLoss, label = "train", color = "r") 
            ax[fold].plot(metricsDf[metricsDf.fold == fold + 1].valEpochLoss, label = "validation", color = "b") 
            ax[fold].set_title("Fold " + str(fold + 1))
            
            # Remove surrounding frame
            ax[fold].spines["left"].set_visible(False) 
            ax[fold].spines["right"].set_visible(False) 
            ax[fold].spines["top"].set_visible(False) 
            ax[fold].spines["bottom"].set_visible(False)
            ax[fold].legend(loc = "upper right") if i == 0 else None
            plt.savefig(join("metrics", "EpochLosses.png"), bbox_inches = "tight", pad_inches = 0, dpi = 200)
            

####################################### Training begins #######################################
# Determine which epoch in "metricsDf" yielded the lowest mean validation loss across the k folds        
meanValLossPerEpoch = metricsDf.groupby("epoch")["valEpochLoss"].mean()
bestEpoch           = meanValLossPerEpoch.idxmin() # 'bestEpoch' is NOT zero-indexed (i.e. first epoch is 1, not 0)

print(f"\nEpoch {bestEpoch} yielded the lowest mean val loss across {args.kFolds} folds: {meanValLossPerEpoch.min():.6f}")        
print("\n------------------------------------------------------------")
print(f"Train on the full dataset for {args.numEpochs} epochs but stop at epoch {bestEpoch}")
print("------------------------------------------------------------")        

# Initialise dataloader for the full dataset, pretrained ResNet-18 model, Adam optimiser and cosine annealing learning rate scheduler
trainSampler        = WeightedRandomSampler(classWeights, len(classWeights), replacement = True)
trainDataLoader     = DataLoader(fullDataset, 
                                 batch_size  = args.batchSize, 
                                 num_workers = args.numWorkers, 
                                 sampler     = trainSampler)

# Initialise fusionet (containing two pretrained ResNet-18 models and a fully connected layer fusing outputs from these models)
model               = fusionet(activationInFCL = args.activationInFCL, 
                               fusion          = args.fusion)
model.to(device)
model               = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model    
optimiser           = torch.optim.Adam(model.parameters(), 
                                       lr           = args.lr, 
                                       weight_decay = args.weightDecay)
LRscheduler         = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.numEpochs)

# Start training: stop and save weights from the "bestEpoch" epoch
for i in trange(0, bestEpoch, desc = "Total number of training epochs") :
    trainEpochLoss, trainEpochMAE = np.round(trainStep(trainLoader = trainDataLoader, 
                                                       useAMP      = True, 
                                                       getOCT      = args.fusion, 
                                                       model       = model, 
                                                       optimiser   = optimiser, 
                                                       lossFn      = lossFn, 
                                                       LRscheduler = LRscheduler, 
                                                       device      = device), 
                                             3)
    print("\nTrain weighted MSE & MAE: " + str(trainEpochLoss) + " & " + str(trainEpochMAE))
    torch.save(model.state_dict(), join(args.weightsDir, "bestEpochWeights.pth"))


