#!/usr/bin/env python
"""
This is an executable script for applying the trained "fusionet" at inference time

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024
"""

import os
import argparse
import pandas as pd
from collections import defaultdict
import cv2 as cv
import numpy as np
from tqdm import tqdm
from model import trainedModel
import torch
join   = os.path.join
root   = os.path.abspath(join(os.getcwd(), "..", "..")) # Project parent directory
device = "cuda" if torch.cuda.is_available() else "mps"


#################################### Setting parameters #####################################
parser = argparse.ArgumentParser(description = "Retinal equivalent refraction prediction")
parser.add_argument("--dataFilePath", 
                    help    = "Full path to the csv file containing the names of input images (image name column must be called 'name')",
                    type    = str) 
parser.add_argument("--outputFilePath", 
                    help    = "Full path to the output file",
                    type    = str, 
                    default = join(root, "output", "RERtestPMprediction.csv"))
parser.add_argument("--weightsDir", 
                    help    = "Name of directory where model checkpoint (weights) is saved",
                    type    = str, 
                    default = "weights")
parser.add_argument("--imageDir", 
                    help    = "Full path to the directory where input images are saved (set to 'None' if 'useRAP' is True",
                    type    = str, 
                    default = join(os.path.dirname(root), "UKB_PM", "images", "full"))
parser.add_argument("--TTA", 
                    help    = "Call if test-time augmentation is desired", 
                    action  = "store_true") 
parser.add_argument("--noPreprocess", 
                    help    = "Input images are NOT preprocessed if called (recommendation: do not call unless the images are already preprocessed)", 
                    action  = "store_false") 
args = parser.parse_args()


###################################### Setup #######################################
# Load trained model 
model = trainedModel(checkpointName = "bestEpochWeights.pth")
model.load(dirPath         = args.weightsDir, 
           activationInFCL = False)

# Get image names
if args.dataFilePath is not None:
    data  = pd.read_csv(args.dataFilePath, low_memory = False)
    names = data.name
else:
    names = [i for i in os.listdir(args.imageDir) if i.endswith(".png") or i.endswith(".jpg") or i.endswith(".jpeg")]

# Create an empty default dictionary to store training & validation metrics
metrics   = defaultdict(list)    

################################ Start test loop ################################    
for j, name in tqdm(enumerate(names)):
    
    try:
        metrics["name"].append(name)
        
        # Read fundus
        fundus = cv.cvtColor(cv.imread(join(args.imageDir, name)), cv.COLOR_BGR2RGB)               
        
        # Predict SER 
        with torch.no_grad():
            if args.TTA:
                predSER, predSER_TTA = model.predict(fundus            = fundus, 
                                                     foveaScan         = None,
                                                     preprocess        = args.noPreprocess, 
                                                     fundusCroppedSize = None,
                                                     TTA               = args.TTA)
                predSER              = predSER.flatten().cpu()
                predSER_TTA          = predSER_TTA.flatten().cpu()
                metrics["predSER"].append(predSER.detach().numpy()[0])                
                metrics["predSER_TTA"].append(predSER_TTA.detach().numpy()[0])  

            else:
                predSER = model.predict(fundus            = fundus, 
                                        foveaScan         = None,
                                        preprocess        = args.noPreprocess, 
                                        fundusCroppedSize = None,
                                        TTA               = args.TTA)
                predSER = predSER.flatten().cpu()
                metrics["predSER"].append(predSER.detach().numpy()[0])            
                
        # Save prediction
        pd.DataFrame.from_dict(metrics).to_csv(args.outputFilePath, index = False)  
    
    except: 
        print("Failed to process", name)
