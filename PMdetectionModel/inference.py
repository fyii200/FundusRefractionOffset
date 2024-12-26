import os
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset import fundusDataset
from model import trainedModel
from tqdm import tqdm
path   = os.path
join   = os.path.join
root   = path.abspath(join(os.getcwd(), "..", ".."))     # Project parent directory
device = "cuda" if torch.cuda.is_available() else "mps"

########################################## SETTINGS ###########################################
parser = argparse.ArgumentParser(description = "Pathologic myopia detection")
parser.add_argument("--dataFile", 
                    help    = "Full path to the csv file containing names of input images",
                    type    = str, 
                    default = join(root, "data", "retinalRefractionGap_normalReferencePrelim.csv")) 
parser.add_argument("--weightsDir", 
                    help    = "Full path to the directory where model checkpoint (weights) is saved",
                    type    = str, 
                    default = join(root, "code", "PMdetectionModel", "weights"))
parser.add_argument("--imageDir", 
                    help    = "Full path to the directory where input images are saved ('None' if 'useRAP' is True)",
                    type    = str, 
                    default = "None")
parser.add_argument("--useRAP", 
                    help    = "Set to True if using UK Biobank's Research Analysis Platform (False if running locally)", 
                    action  = "store_true") # False if not called
parser.add_argument("--batchSize", 
                    help    = "batch size for dataloader; default is 16",
                    type    = int, 
                    default = 16)
parser.add_argument("--numWorkers", 
                    help    = "number of workers for dataloader; default is 4",
                    type    = int, 
                    default = 4)
parser.add_argument("--optimalThreshold", 
                    help    = "Optimal threshold for binary classification during validation/inference (if -1 then the cutoff will be set to 0.5).",
                    type    = float, 
                    default = 0.15)
parser.add_argument("--outputFile", 
                    help    = "Full path to the output file containing the predicted pathologic myopia labels",
                    type    = str, 
                    default = join(root, "output", "PMdetection.csv"))
parser.add_argument("--TTA",
                    help    = "Apply test-time augmentation at inference time",
                    action  = "store_true") # False if not called

args = parser.parse_args()

# Dataset
d = pd.read_csv(args.dataFile, low_memory = False)
print(str(len(d)) + " images in total")

# Load trained model
model = trainedModel(checkpointName = "bestEpochWeights.pth")
model.load(weightsDir = args.weightsDir, numClasses = 2)

# Initialise dataloader
fullDataset = fundusDataset(imageDir   = args.imageDir, 
                            dataFrame  = d, 
                            numClasses = 2, 
                            useRAP     = args.useRAP, 
                            inference  = True, 
                            resizedDim = (512, 512))

dataloader  = DataLoader(fullDataset, 
                         batch_size    = args.batchSize, 
                         num_workers   = args.numWorkers)

# Start inferencing
metrics = defaultdict(list)
for name, image in tqdm(dataloader):
    image = image.to(device)
    finalScores, predictedProb, predictedClass = model.predict(image, 
                                                               TTA              = args.TTA, 
                                                               optimalThreshold = args.optimalThreshold)
    
    [metrics["name"].append(i)           for i in name                                 ]
    [metrics["predictedProb"].append(i)  for i in predictedProb.cpu().detach().numpy() ]
    [metrics["predictedClass"].append(i) for i in predictedClass.cpu().detach().numpy()]
    pd.DataFrame(metrics).to_csv(args.outputFile, index = False)    
