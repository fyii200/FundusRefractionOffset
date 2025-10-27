"""
This script contains the custom dataset class

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset
join = os.path.join

class fundusDataset(Dataset):

    def __init__(self, dataFrame, imageDir, useRAP = False, inference = False, croppedSize = (1400,1400), resizedDim = (256,256), numClasses = 2):
        
        """
        Args:
             dataFrame (2D tabular)       : Pandas dataframe with image names and groundtruth annotation.
             imageDir (str)               : Full path to the directory where input images are saved (set to None if "useRAP" is True).
             useRAP (bool)                : Set to True if using UK Biobank's Research Analysis Platform (False if running locally).
             inference (bool)             : Set to False during training/validation to get groundtruth label corresponding to each image.
             croppedSize (tuple of int)   : Desired size of the centre crop for fundus image to remove the surrounding black border.
             resizedDim (tuple of int)    : Desired input size of fundus images.
             numClasses (int)             : Number of classes.
        Out:
             imageName (str)              : Image names of length [B].
             image (4D tensor)            : Batch of images of shape [B,H,W,C].
             PMcategoryOneHot (2D tensor) : One-hot encoded labels of shape [B,numClasses].
             PMbinary (1D tensor)         : 0. if no PM and 1. if PM is present.
        """
        self.dataFrame   = dataFrame
        self.imageDir    = imageDir
        self.useRAP      = useRAP
        self.inference   = inference
        self.numClasses  = numClasses
        self.crop        = CenterCrop(croppedSize)
        self.resizedDim  = resizedDim          

    def __len__(self):
        
        return len(self.dataFrame)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()    
         
        imageName = self.dataFrame.name.iloc[idx]

        if self.useRAP:
            if self.dataFrame.eye.iloc[idx] == 'RE':
                self.imageDir = join(os.sep, "mnt", "project", "Bulk", 
                                     "Retinal Optical Coherence Tomography", "Fundus (right)", str(imageName[0:2]))
            elif self.dataFrame.eye.iloc[idx] == 'LE':
                self.imageDir = join(os.sep, "mnt", "project", "Bulk", 
                                     "Retinal Optical Coherence Tomography", "Fundus (left)", str(imageName[0:2]))
                
            if os.path.exists(join(self.imageDir, imageName)):
                imgPath = join(self.imageDir, imageName) 
                # Fall back to the first image if an error occurs due to whatever reason (e.g. corrupted file)
            else:
                imgPath = join(os.sep, "mnt", "project", "Bulk", 
                               "Retinal Optical Coherence Tomography", "Fundus (left)", "10", "1000071_21015_0_0.png")
                print("Fall back to the first example because current file may be corrupted!")
        else:
            imgPath = join(self.imageDir, imageName)
        
        # Read image
        image   = cv.imread(imgPath)
            
        # Convert from BGR to RGB
        image   = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Centre crop image to remove surrounding black border and resize to resizeDim pixels
        image   = self.crop(torch.as_tensor(image).moveaxis(2,0)).moveaxis(0,2).detach().numpy() 
        image   = cv.resize(image, self.resizedDim)      

        # Preprocess input images: normalise to [0-1] range and reshaped from [H,W,C] into [C,H,W]
        image   = torch.tensor(np.uint8(image)).moveaxis(2, 0) 
        image   = image/255
         
        # Return output    
        if self.inference:
            return imageName, image
        else:
            # groundtruth label (binary)
            PMbinary = self.dataFrame.PMbinary_eyeLevel.iloc[idx]
            
            # One-hot encode groundtruth label
            PMbinary = torch.tensor(np.float32(PMbinary)) 
            PMcategoryOneHot = torch.zeros(self.numClasses)
            PMcategoryOneHot[np.uint8(PMbinary)] = 1
            return imageName, image, PMcategoryOneHot, PMbinary 
    
    

    
    
    
    
    
    
