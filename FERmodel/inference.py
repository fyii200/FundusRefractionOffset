#!/usr/bin/env python
"""
This is an executable script for applying the trained model to internal (UK Biobank) dataset at inference time

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024
"""

import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from dataset import DatasetGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import trainedModel
import torch
join   = os.path.join
root   = os.path.abspath(join(os.getcwd(), "..", "..")) # Project parent directory
device = "cuda" if torch.cuda.is_available() else "cpu"


#################################### Setting parameters #####################################
parser = argparse.ArgumentParser(description = "Retinal equivalent refraction prediction")
parser.add_argument("--dataFile", 
                    help    = "Full path to the csv file containing names of input images (image name column must be called 'name')",
                    type    = str, 
                    default = join(root, "data", "retinalRefractionGap_test_new.csv")) 
parser.add_argument("--testWhatDataset", 
                    help    = "Which dataset is used for inference?",
                    type    = str, 
                    default = "test")
parser.add_argument("--weightsDir", 
                    help    = "Name of directory where model checkpoint (weights) is saved",
                    type    = str, 
                    default = "weights")
parser.add_argument("--imageDir", 
                    help    = "Full path to the directory where input images are saved (set to 'None' if 'useRAP' is True",
                    type    = str, 
                    default = "None")
parser.add_argument("--numWorkers", 
                    help    = "Number of workers for dataloader; default is 8",
                    type    = int, 
                    default = 8)
parser.add_argument("--fusion", 
                    help    = "Set to True if fundus images and OCT scans are used as inputs; False is only fundus images are used", 
                    action  = "store_true") # False if not called
parser.add_argument("--useRAP", 
                    help    = "Set to True if using UK Biobank's Research Analysis Platform (False if running locally)", 
                    action  = "store_true") # False if not called
parser.add_argument("--TTA", 
                    help    = "Call if test-time augmentation is desired", 
                    action  = "store_true") 
parser.add_argument("--activationInFCL", 
                    help    = "Whether tanH activation function should be added to the fully connected layers",
                    action  = "store_true") # False if not called
args = parser.parse_args()


###################################### Setup #######################################
# Load trained model 
model = trainedModel(checkpointName = "bestEpochWeights.pth")
model.load(dirPath         = args.weightsDir, 
           activationInFCL = args.activationInFCL)

# Read full dataframe and create subset all images
data  = pd.read_csv(args.dataFile, low_memory = False)
    
# Initialise test loader
testDataset = DatasetGenerator(dataFrame   = data,
                               imageDir    = args.imageDir,
                               useRAP      = args.useRAP, 
                               train       = False,
                               getOCT      = args.fusion)
testLoader  = DataLoader(testDataset, 
                         batch_size  = 1, 
                         num_workers = args.numWorkers, 
                         pin_memory  = True)
# MAE loss function
MAElossFn = torch.nn.L1Loss(size_average = None, reduce = None)   

# MSE loss function
MSElossFn = torch.nn.MSELoss(size_average = None, reduce = None) 

# Create an empty default dictionary to store training & validation metrics
metrics   = defaultdict(list)

# Progress bar
testIter  = tqdm(iter(testLoader), desc = "Test batch")
    
    
################################ Start test loop ################################    
for j, (name, GT_SER, *image) in enumerate(testIter):
    metrics["name"].append(name[0])
    
    if args.fusion:
        fundus, foveaScan = image
        fundus            = fundus.to(device)                 
        foveaScan         = foveaScan.to(device)
    else:
        fundus            = image[0].to(device)                 
    
    with torch.no_grad():
        # Predict SER 
        if args.TTA:
            predictedSER, predictedSER_TTA = model.predict(fundus, foveaScan, TTA = True) if args.fusion else model.predict(fundus, None, TTA = True)
            predictedSER_TTA               = predictedSER_TTA.flatten().cpu()
        else:
            predictedSER                   = model.predict(fundus, foveaScan, TTA = False) if args.fusion else model.predict(fundus, None, TTA = False)
            
        predictedSER = predictedSER.flatten().cpu()
        
        # Calculate MAE and MSE
        absoluteDiff = MAElossFn(predictedSER, GT_SER)
        squaredDiff  = MSElossFn(predictedSER, GT_SER)             
        
    ### Save performance metrics ###
    metrics["absoluteDiff"].append(absoluteDiff.item()) 
    metrics["squaredDiff"].append(squaredDiff.item()) 
    metrics["trueSER"].append(GT_SER.detach().numpy()[0])
    metrics["predSER"].append(predictedSER.detach().numpy()[0])
    metrics["predSER_TTA"].append(predictedSER_TTA.detach().numpy()[0]) if args.TTA else None
    pd.DataFrame.from_dict(metrics).to_csv(join(root, "output", "RER" + args.testWhatDataset + "Prediction.csv"), index = False)  
