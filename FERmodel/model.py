"""
This script defines the "fusionet" class, a deep learning model with one ResNet-18 backbone, if "fusion" is False,
or two ResNet-18 backbones fused through two fully connected layers, if "fusion" is True.

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import cv2 as cv
import torch
from torch import nn
import numpy as np
import torchvision.models as models
from torchvision.transforms import RandomHorizontalFlip, CenterCrop, functional, ColorJitter
join = os.path.join

class trainedModel:
    def __init__(self, checkpointName):
        """
        Args:
              checkpointName (str): Name of the file containing the trained weights (.pth).
        """
        
        self.checkpointName  = checkpointName
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.testHorFlip     = RandomHorizontalFlip(p = 1)
        self.cj1             = ColorJitter(brightness = [0.5, 0.5], saturation = [0.5, 0.5])
        self.cj2             = ColorJitter(brightness = [0.8, 0.8], saturation = [0.8, 0.8])        

    def load(self, dirPath, activationInFCL = False):
        """
        Args:
              dirPath (str)          : Path to the weights.
              activationInFCL (bool) : If true, tanH activation is used in the penultimate fully connected layer.    
        """
        
        # Load fusionet and its trained weights.
        self.model     = fusionet(activationInFCL)
        checkpointPath = join(dirPath, self.checkpointName)
        stateDict      = torch.load(checkpointPath, map_location = self.device)
        try:
            # Remove "module." if it is in the keys
            newStateDict = {}
            for k, v in stateDict.items():
                newStateDict[k.replace('module.', '')] = v
        except:
            newStateDict = stateDict
        self.model.load_state_dict(newStateDict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, fundus, foveaScan, preprocess = False, TTA = True, fundusCroppedSize = (1400,1400)):
        """
        Args:
              fundus (4D tensor)               : Input fundus image of shape [B,H,W,C].
              preprocess (boolean)             : True if input needs to be preprocessed. 
              TTA (boolean)                    : True if test-time augmentation. 
              fundusCroppedSize (tuple of int) : Desired size of the centre crop for UK Biobank fundus image (default to
                                                 1400 by 1400 pixels to remove the black/empty border around the image).              
        Out:
              score (int)  : predicted spherical equivalent refraction (SER).
        """
        
        if preprocess:
            # Centre crop fundus image to "croppedSize"
            if fundusCroppedSize != 0:
                crop   = CenterCrop(fundusCroppedSize)
                fundus = crop(torch.as_tensor(fundus).moveaxis(2,0)).moveaxis(0,2).detach().numpy() 
                
            # Crop automatically if "croppedSize" is not provided
            else:
                yNonzero, xNonzero, _  = np.nonzero(fundus)
                fundus                 = fundus[np.min(yNonzero):np.max(yNonzero), np.min(xNonzero):np.max(xNonzero)]            
                
            # Resize input image to 512 by 512 pixels and normalise to [0-1] range
            fundus = cv.resize(fundus, (512,512), interpolation = cv.INTER_AREA)
            fundus = torch.from_numpy(fundus).permute(2,0,1).unsqueeze(0)
            fundus = fundus / torch.max(fundus)
            fundus = fundus.to(self.device, torch.float)

        with torch.no_grad(): 
            oriSER = self.model(fundus, foveaScan)  
            # Predict SER with test-time augmentation    
            if TTA:
                fundusHorFlipped     = self.testHorFlip(fundus)
                fundusCJ1            = self.cj1(fundus)
                fundusCJ2            = self.cj2(fundus)                
                horFlippedSER        = self.model(x1 = fundusHorFlipped, x2 = None)    
                cj1SER               = self.model(x1 = fundusCJ1, x2 = None)    
                cj2SER               = self.model(x1 = fundusCJ2, x2 = None)    
                rotatedSER1          = self.model(functional.rotate(fundus, -3), x2 = None)
                rotatedSER2          = self.model(functional.rotate(fundus, 3), x2 = None)                  
                rotatedSER3          = self.model(functional.rotate(fundus, -5), x2 = None)                  
                rotatedSER4          = self.model(functional.rotate(fundus, 5), x2 = None)                                                  
                # Average predictions
                SER_TTA              = (oriSER + horFlippedSER + cj1SER + cj2SER + rotatedSER1 + rotatedSER2 + rotatedSER3 + rotatedSER4) / 8
                
                return oriSER, SER_TTA
            
            else:
                return oriSER                
        
        
    
    
class ResNet18(nn.Module):
    def __init__(self):
        """
        Initialise a pre-trained ResNet18 model.
        """
        
        super(ResNet18, self).__init__()
        self.resnet    = models.resnet18(pretrained = True)
        num_features   = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=1) # only one output required for regression

    def forward(self, x):
        x = self.resnet(x)
        return x   
    

class fusionet(nn.Module):
    def __init__(self, activationInFCL = False, fusion = False):
        """
        Initialise fusionet, a fusion CNN containing two pre-trained ResNet18 models and a fully connected layer.
        
        Args:
              activationInFCL (bool) : If True TanH activation is added to the fully connected layers (default is False). 
              fusion (bool)          : If True the model takes both fundus images and OCT scans as inputs; 
                                       otherwise, only fundus images as inputs (default is False)
        """
        
        super(fusionet, self).__init__()
        self.activationInFCL  = activationInFCL
        self.fusion           = fusion
        
        # Initialise two Resnet-18 models
        if self.fusion:
            self.fundusnet    = models.resnet18(pretrained = True)
            self.OCTnet       = models.resnet18(pretrained = True)
            # Remove the last fully connected layer
            self.fundusnet.fc = nn.Identity()
            self.OCTnet.fc    = nn.Identity()
            # Output size of each model is 512, so the input size of the first fully connected layer is 512*2
            self.FCL1         = nn.Linear(in_features = 512 + 512, out_features = 512)
        
        # Initialise one Resnet-18 model
        else:
            self.fundusnet    = models.resnet18(pretrained = True)
            # Remove the last fully connected layer
            self.fundusnet.fc = nn.Identity()
            self.FCL1         = nn.Linear(in_features = 512, out_features = 512)
        
        # Single-output regression, so one out_features in the final fully connected layer
        self.FCL2 = nn.Linear(in_features = 512, out_features = 1)
        

    def forward(self, x1, x2 = None):
        """
        Args:
              x1 (4D tensor)            : Input fundus image of shape [B,C,H,W].
              x2 (4D tensor)            : Input OCT scan of shape [B,C,H,W].
        Out:
              x (float)                 : Model output (predicted spherical equivalent refraction).
        """
        
        # Forward pass fundus images through fundusnet
        x = self.fundusnet(x1)  
        
        # Forward pass OCT scans through OCTnet and concatenate outputs from both models
        if self.fusion:
            x2 = self.OCTnet(x2) 
            x  = torch.cat((x, x2), dim = 1)   
            
        # 2 fully connected layers
        x      = self.FCL1(x)
        x      = torch.tanh(x) if self.activationInFCL else x
        x      = self.FCL2(x)

        return x           
