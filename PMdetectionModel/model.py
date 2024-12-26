"""
This script is used to initialise ResNet18 and the trained model used at inference time.

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import cv2 as cv
import torchvision
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms.functional import rotate
from torch.nn.functional import softmax
join = os.path.join

class trainedModel:
    def __init__(self, checkpointName = "bestEpochWeights.pth"):
        """
        Args:
              checkpoint (str): Name of the file containing the trained weights (.pth).
        """
        
        self.checkpointName = checkpointName
        self.testHorFlip    = transforms.RandomHorizontalFlip(p = 1)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "mps")

    def load(self, weightsDir, numClasses):
        """
        Args:
              weightsDir (str) : Path to the weights.
              numClasses (int) : Number of groundtruth classes (2 for binary classification).
        """
        
        # Load ResNet18 and its trained weights.
        self.model      = ResNet18(imageNet = False, numClasses = numClasses)
        checkpointPath  = join(weightsDir, self.checkpointName)
        stateDict       = torch.load(checkpointPath, map_location = self.device)
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

    def predict(self, image, TTA = True, optimalThreshold = -1):
        """
        Args:
              image (4D tensor)        : Input image of shape [B,C,H,W].
              TTA (bool)               : Test-time agmentations (horizontal flip & rotation) are applied if True.
              optimalThreshold (float) : Optimal threshold for binary classification (if -1 then the cutoff will be set to 0.5).
        Out:
              finalScores (float)      : Decision scores of shape [B,numClasses] output by the model.
              predictedProb (float)    : Predicted probability (1D array) of the positive class (i.e. pathologic myopia is present).
              predictedClass (int)     : Predicted pathologic myopia class (1D array: 1 if present and 0 if absent).
        """
        with torch.no_grad(): 
            # Predict decision scores (one for each class) from the original image and 10 other variants of the same image (if "TTA" is True).
            scores               = self.model(image.to(self.device))                  
            if TTA: 
                scoresHorFlipped = self.model(self.testHorFlip(image))         
                scoresRotated1   = self.model(rotate(image, -5))
                scoresRotated2   = self.model(rotate(image, 5))
                # Take the average of the decision scores predicted from the original image and its variants as the final decision scores.
                scoresSummed     = (scores + 
                                    scoresHorFlipped + 
                                    scoresRotated1   + 
                                    scoresRotated2)
                finalScores      = scoresSummed / 4
            else:
                finalScores      = scores

       # Turn model output into predicted probability of having pathologic myopia and binary label using 0.5 or "optimalThreshold" (if specified) as cutoff
        predictedProb        = softmax(finalScores, dim = 1)[:,1] 
        optimalThreshold     = 0.5 if optimalThreshold == -1 else optimalThreshold  
        predictedClass       = (predictedProb > optimalThreshold).int()        
        return finalScores, predictedProb, predictedClass
    
    
class ResNet18(nn.Module):
    def __init__(self, customWeightsPath = None, imageNet = True, numClasses = 2):
        """
        Initialise a pre-trained ResNet18 model.
        
        Args:
             customWeightsPath : If specified (not None), model will be initialised using weights found in 
                                 the file path and overwrite "imageNet" flag.
             imageNet          : If True, model will be initialised using weights pretrained on ImageNet. 
             numClasses (int)  : Number of groundtruth classes (2 for binary classification).
        """
        super(ResNet18, self).__init__()
        
        try:
            self.resnet = resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT if imageNet else None)
        except:
            self.resnet = resnet18(imageNet)
        numInFeatures   = self.resnet.fc.in_features
        self.resnet.fc  = nn.Linear(in_features = numInFeatures, out_features = numClasses)
        
        # Load custom weights if provided
        if customWeightsPath:
            print("Model initialised with weights pretrained on MMAC dataset")
            stateDict = torch.load(customWeightsPath, map_location = "cuda" if torch.cuda.is_available() else "mps")
            # Remove "resnet." prefix from each key if it exists
            newStateDict = {}
            for k, v in stateDict.items():
                newKey               = k.replace("resnet.", "") 
                newStateDict[newKey] = v
            # Ensure the size of the fully connected layer is consistent with "numClasses"
            newStateDict['fc.weight'] = newStateDict['fc.weight'][0:numClasses,:]
            newStateDict['fc.bias']   = newStateDict['fc.bias'][0:numClasses]    
            # Load pretrained weights
            self.resnet.load_state_dict(newStateDict, strict = False)

    def forward(self, x):
        x = self.resnet(x)
        return x   
    
    
