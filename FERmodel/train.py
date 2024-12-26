#!/usr/bin/env python
"""
This is an executable script for training a deep learning model called "fusionet" to predict 
"retinal equivalent refraction" from fundus images and OCT B-scans

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024
"""

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

import os
import argparse
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from model import fusionet
from dataset import DatasetGenerator
from torch.utils.data import Dataset, DataLoader
from utils import trainValTestSplit, trainStep, valStep, weightedMSE
from tqdm import trange
import matplotlib
matplotlib.use("Agg") # turn off plot display
import matplotlib.pyplot as plt

join   = os.path.join
plt    = matplotlib.pyplot
device = "cuda" if torch.cuda.is_available() else "cpu"
root   = os.path.abspath(join(os.getcwd(), "../.."))     # Project parent directory

os.environ["NCCL_DEBUG"]      = "OFF"
os.environ["NCCL_IB_DISABLE"] = "1"


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
parser.add_argument("--numEpochs", 
                    help    = "total number of epochs; default is 20",
                    type    = int, 
                    default = 20)
parser.add_argument("--MSEweightFactor", 
                    help    = "Weight factor applied to the weight used in the loss function (larger value = larger penalty)",
                    type    = float, 
                    default = 1.2)
parser.add_argument("--fusion", 
                    help    = "Set to True if fundus images and OCT scans are used as inputs; False is only fundus images are used", 
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
data = pd.read_csv(join(root, "data", "retinalRefractionGap_train.csv"), low_memory = False)
print(str(len(data)) + " images")

# Create train set, train data sampler (helps oversample minority classes with replacement during training) and validation set
trainSet, trainSampler, valSet = trainValTestSplit(data, trainProportion = 0.8, createTestSet = False)

# Initialise fusionet
fusionet = fusionet(activationInFCL = args.activationInFCL, fusion = args.fusion)
fusionet.to(device)

# parallelise data across multiple GPUs if there's more than 1 GPU
fusionet = torch.nn.DataParallel(fusionet) if torch.cuda.device_count() > 1 else fusionet

# Initialise train and validation dataloaders
trainDataset = DatasetGenerator(dataFrame         = trainSet,
                                imageDir          = args.imageDir,
                                useRAP            = args.useRAP, 
                                train             = True, 
                                getOCT            = args.fusion,                                
                                fundusCroppedSize = (args.fundusCroppedSize,args.fundusCroppedSize), 
                                resizedDim        = (args.resizedDim,args.resizedDim))
valDataset   = DatasetGenerator(dataFrame         = valSet,
                                imageDir          = args.imageDir,
                                useRAP            = args.useRAP,                                   
                                train             = False, 
                                getOCT            = args.fusion,                                
                                fundusCroppedSize = (args.fundusCroppedSize,args.fundusCroppedSize), 
                                resizedDim        = (args.resizedDim,args.resizedDim))
trainDataLoader = DataLoader(trainDataset, 
                             batch_size       = args.batchSize, 
                             num_workers      = args.numWorkers,
                             pin_memory       = True,
                             sampler          = trainSampler)
valDataLoader   = DataLoader(valDataset, 
                             batch_size       = args.batchSize, 
                             num_workers      = args.numWorkers, 
                             pin_memory       = True)

# Save train and validation image names
pd.DataFrame(trainSet.name).to_csv(join("imageNames", "trainNames.csv"))  
pd.DataFrame(valSet.name).to_csv(join("imageNames", "valNames.csv"))  

# Initialise Adam optimiser with tuned lr, weightDecay, betas & eps values
optim = torch.optim.Adam(fusionet.parameters(),
                         lr           = args.lr,
                         weight_decay = args.weightDecay)

# Initialise cosine annealing scheduler
LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = args.numEpochs)

# Mean squared error loss function 
trainLossFn = weightedMSE(args.MSEweightFactor) # weighted if args.MSEweightFactor is > 0
valLossFn   = weightedMSE(args.MSEweightFactor) # weighted if args.MSEweightFactor is > 0                   

# Create an empty default dictionary to store training & validation metrics
metrics     = defaultdict(list)


####################################### Training begins #######################################
# Placeholder to store smallest (best) validation loss
bestValLoss = np.inf   

# Train for "args.numEpochs" epochs
for i in trange(0, args.numEpochs, desc = "Epochs") :    
    metrics["epoch"].append(i+1)
    
    ### Training step ###
    trainEpochLoss = np.round(trainStep(trainDataLoader, True, args.fusion, fusionet, optim, trainLossFn, LRscheduler, device), 4)
    metrics["trainEpochLoss"].append(trainEpochLoss)
    print("\n" + str(trainEpochLoss))
    
    ### Validation step ###
    valEpochLoss = np.round(valStep(valDataLoader, fusionet, args.fusion, valLossFn, device), 4)
    metrics["valEpochLoss"].append(valEpochLoss)
    print("\n" + str(valEpochLoss))     
    
    print("^^^^^^^", "Epoch " + str(i+1), " done ^^^^^^^")
    
    ### Save best model parameters ###
    if valEpochLoss < bestValLoss:          
        bestValLoss = valEpochLoss
        torch.save(fusionet.state_dict(), join(args.weightsDir, "bestEpochWeights.pth"))
        
    ### Save checkpoint ###    
    torch.save({
        "epoch"             : i+1,
        "model"             : fusionet.state_dict(),
        "optimiser"         : optim.state_dict(),
        "LRscheduler"       : LRscheduler.state_dict(),
        "lossType"          : "weightedMSE" if args.MSEweightFactor != 0 else "nonweightedMSE",
        "trainEpochLoss"    : trainEpochLoss,
        "valEpochLoss"      : valEpochLoss,
        "bestValLoss"       : bestValLoss,
        "MSEweightFactor"   : args.MSEweightFactor,
        "fundusCroppedSize" : args.fundusCroppedSize,
        "resizedDim"        : args.resizedDim,
        "batchSize"         : args.batchSize
    }, join(args.weightsDir, "checkpointEpoch" + str(i+1) + ".pth"))        

    ### Save performance metrics ###
    pd.DataFrame.from_dict(metrics).to_csv(join("metrics", "trainValidationLosses.csv"), index=False)
    
    ### Plot loss vs epoch ###
    if args.plotLosses:
        plt.plot(metrics["trainEpochLoss"], label = "train", color = "r") 
        plt.plot(metrics["valEpochLoss"], label = "validation", color = "b")
        plt.title("Loss vs Epoch")
        plt.legend(loc = "upper right") if i == 0 else None
        plt.savefig(join("metrics", "epochLosses.png"))
        
        
