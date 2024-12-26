"""
This script contains custom helper functions

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import numpy as np
import torch
import pandas as pd
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import rotate
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from tqdm import tqdm
join = os.path.join



#################################### Regular augmentations #####################################
class augmentation:
    def __init__(self, angles):
        """
        Args:
              angles (list): List of angles to be randomly sampled from.
        """
            
        self.angles = angles

    def __call__(self, inputImage, randomRotate = True):
        """
        Args:
              inputImage (3D or 4D tensor)  : Expect to have [...,C,H,W] shape.
              randomRotate (bool)           : Random rotation if True. 
        
        Out: 
              outputImage (3D or 4D tensor) : Augmented image (3D tensor) or images (4D tensor).
        """
        
        # Initialise random brightness & saturation jitter;
        # jitter factor randomly chosen from 0.5 to 1.5
        cj = ColorJitter(brightness = 0.5, saturation = 0.5)
        
        # Initialise random horizontal flip
        Hflip = RandomHorizontalFlip(p = 0.5)
        
        # Randomly sample an angle from a predefined list of angles
        unif  = torch.ones(len(self.angles))
        idx   = unif.multinomial(1)
        angle = self.angles[idx]
        
        # Random rotation, followed by horizontal flip and brightness/saturation jitter
        if randomRotate:
            outputImage = cj(Hflip(rotate(inputImage, angle)))
        else:
            outputImage = cj(Hflip(inputImage))
        
        return outputImage
    
    

######################### Load a saved checkpoint to resume training ###########################
def loadCheckpoint(weightsDir, epoch, model, optimiser, LRscheduler, resetLR):
    """
    Args:
          weightsDir (str)            : Path to the directory where training checkpoints (model weights) are saved.
          epoch (int)                 : Epoch associated with the checkpoint to be loaded. 
          model (torchvision.models)  : Most recently trained model.
          optimiser (torch.optim)     : Initialised optimising algorithm.
          LRscheduler (torch.optim)   : Initialised learning rate scheduler.
          resetLR (float)             : If specified, initial learning rate used by the optimiser will be reset to this value.
    """    
    
    checkpoint = torch.load(join(weightsDir, "checkpointEpoch" + str(epoch) + ".pth") )

    model.load_state_dict(checkpoint["model"])
    
    #######################################################################
    #### Reset learning rate stored in the optimiser to a desired value ###
    #######################################################################    
    if resetLR != "None":
        print("Reset learning rate to " + str(resetLR) + "!")
        checkpoint["optimiser"]["param_groups"][0]["lr"] = resetLR
        checkpoint["LRscheduler"]["last_epoch"]          = 0
        checkpoint["LRscheduler"]["_step_count"]         = 1
        checkpoint["LRscheduler"]["_last_lr"]            = [resetLR]
    ####################################################################### 
    
    optimiser.load_state_dict(checkpoint["optimiser"])
    LRscheduler.load_state_dict(checkpoint["LRscheduler"])

    return (model, 
            optimiser, 
            LRscheduler, 
            checkpoint["epoch"], 
            checkpoint["lossType"], 
            checkpoint["bestValLoss"],
            checkpoint["MSEweightFactor"], 
            checkpoint["fundusCroppedSize"], 
            checkpoint["resizedDim"], 
            checkpoint["batchSize"])  



################################# Train-validation-test split ##################################
def trainValTestSplit(data, trainProportion = 0.8, seed = 1, createTestSet = False):
    """
    Args:
          data (dataframe)              : Pandas dataframe (with participant information).
          trainProportion (float)       : Ranges from 0. to 1. (proportion of data prescribed as training set).
          seed (integer)                : Seed for numpy random number generator.
          createTestSet (Bool)          : If True, test set will be created along with train and validation sets.
    Out:
          trainSet (dataframe)          : Train set.
          trainSampler (torch Sampler)  : Data sampler for train set (oversampling minority classes with replacement during training).
          valSet (dataframe)            : Validation set.
          testSet (torch Sampler)       : Test set (if createTestSet is True).
    """
    
    # Create data indices for the full dataset
    datasetSize = len(data)
    indices     = list(range(datasetSize))
    
    # Shuffle indices (full dataset)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Split indices corresponding to the full dataset into training set and validation set
    trainSplit                 = int(np.ceil(trainProportion * datasetSize))
    (trainIndices, valIndices) = indices[:trainSplit], indices[trainSplit:]
    
    # Create train set and validation set
    trainSet = data.iloc[trainIndices].copy()
    valSet   = data.iloc[valIndices].copy()
    
    # Create train sampler for oversampling minority classes (high myopia) with replacement during training
    trainSet["SERclass"]                        = [2] * len(trainSet) 
    trainSet.loc[trainSet.SER < -8, "SERclass"] = 3                   
    trainSet.loc[trainSet.SER > -5, "SERclass"] = 1                   
    trainClassProp                              = trainSet["SERclass"].value_counts(normalize = True)          
    trainClassWeights                           = [1 / trainClassProp[i] for i in trainSet["SERclass"]]          
    trainSampler                                = WeightedRandomSampler(weights     = trainClassWeights, 
                                                                        num_samples = len(trainClassWeights), 
                                                                        replacement = True)
    print("Train set: "                                                            + 
          str(round(trainClassProp.values[0]*100)) + "%" + " SER > -5 | "          + 
          str(round(trainClassProp.values[1]*100)) + "%" + " SER b/w -5 & -8 | "   +
          str(round(trainClassProp.values[2]*100)) + "%" + " SER < -8 " )

    
    # Split indices corresponding to the validation set further into a validation set and a test set if "createTestSet" is True
    if createTestSet:
        testSplit               = int(np.ceil(len(valIndices)/2))
        valIndices, testIndices = valIndices[:testSplit], valIndices[testSplit:]
        valSet                  = data.iloc[valIndices].copy()                       
        testSet                 = data.iloc[testIndices].copy()                      
        return trainSet, trainSampler, valSet, testSet 
    else:
        return trainSet, trainSampler, valSet
        


########################## Train step (one step completes one epoch) ###########################
def trainStep(trainLoader, useAMP, getOCT, model, optimiser, lossFn, LRscheduler, device):
    """
    Run one training step (i.e. one epoch)
    
    Args:
          trainLoader (torch.utils.data.DataLoader)  : Initialised dataloader for the training set.
          useAMP (bool)                              : Average mixed precision is activated to speed up runtime 
                                                       and reduce memory footprint if set to True.
          getOCT (bool)                              : Get OCT scans if True (default is False).                                           
          model (torchvision.models)                 : Initialised regression model (default: ResNet18).
          optimiser (torch.optim)                    : Initialised optimising algorithm (default: ADAM).
          lossFn (torch.nn)                          : Initialised loss function (default: mean absolute error).
          LRscheduler (torch.optim.lr_scheduler)     : Initialised learning rate scheduler (default: cosine annealing).
          device (str)                               : "cuda" (GPU) or "cpu".
          
    Out:
          trainEpochLoss (tensor float)              : Training loss in a given training epoch.
    """
    
    # Train mode    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = useAMP)  
    
    # Placeholder for batch losses and MAE summed across a training epoch
    totalBatchTrainLoss = 0. 
    totalBatchTrainMAE  = 0.
    
    # Progress bar
    trainBatchIter = tqdm(iter(trainLoader), desc = "Training batch")
    for j, (_, GT_SER, *image) in enumerate(trainBatchIter):
        GT_SER     = GT_SER.to(device, torch.float32)
        
        if getOCT:
            fundus, foveaScan = image
            fundus            = fundus.to(device)
            foveaScan         = foveaScan.to(device)
        else:
            fundus            = image[0].to(device)
            
        # Predict SER (forward pass)
        with torch.autocast(device_type = device, dtype = torch.float16, enabled = useAMP):
            predictedSER = model(x1 = fundus, x2 = foveaScan) if getOCT else model(x1 = fundus, x2 = None)           
            predictedSER = predictedSER.flatten()  
            loss         = lossFn(predictedSER, GT_SER)  
            MAE          = torch.mean(abs(predictedSER - GT_SER))
            
        # Backprop and update parameters
        scaler.scale(loss).backward()           # "scale" multiplies loss by current scale factor (prevent gradients from flushing to 0)
        scaler.step(optimiser)                  # Unscales gradients and calls optimiser.step()
        scaler.update()                         # Updates scaler’s scale factor for the next iteration
        optimiser.zero_grad(set_to_none = True)
        totalBatchTrainLoss += loss.item() 
        totalBatchTrainMAE  += MAE.item() 
    
    # Update learning rate
    LRscheduler.step()
    
    # Compute and save epoch loss
    trainEpochLoss = totalBatchTrainLoss / (j+1)
    trainEpochMAE = totalBatchTrainMAE / (j+1)
    
    return trainEpochLoss, trainEpochMAE



######################## Validation step (one step completes one epoch) ########################
def valStep(valLoader, model, getOCT, lossFn, device):
    """
    Run one validation step (i.e. one epoch) — always preceded by a training step
    
    Args:
          valLoader (torch.utils.data.DataLoader)    : Initialised dataloader for the validation set.
          model (torchvision.models)                 : Initialised regression model (default: ResNet18).
          getOCT (bool)                              : Get OCT scans if True (default is False).                                           
          lossFn (torch.nn)                          : Initialised loss function (default: non-weighted MSE).
          device (str)                               : "cuda" (GPU) or "cpu".
          
    Out:
          valEpochLoss (tensor float)                : Validation loss in current training epoch.
          valMAE (tensor float)                      : Validation mean absolute error.
    """
    
    # Evaluation mode
    model.eval()
    
    # Placeholder for batch losses and MAE summed across a validation epoch
    totalBatchValLoss = 0.
    totalBatchValMAE  = 0.
    
    # Progress bar
    valBatchIter  = tqdm(iter(valLoader), desc = "Validation batch")
    for j, (_, GT_SER, *image) in enumerate(valBatchIter):
        GT_SER    = GT_SER.to(device, torch.float32)

        if getOCT:
            fundus, foveaScan = image
            fundus            = fundus.to(device)                 
            foveaScan         = foveaScan.to(device)                 
        else:
            fundus            = image[0].to(device)                 
            
        with torch.no_grad():
            # Predict SER
            predictedSER      = model(x1 = fundus, x2 = foveaScan) if getOCT else model(x1 = fundus, x2 = None)          
            predictedSER      = predictedSER.flatten()
            
            # Calculate validation loss
            loss               = lossFn(predictedSER, GT_SER) 
            MAE                = torch.mean(abs(predictedSER - GT_SER))
            totalBatchValLoss += loss.item() 
            totalBatchValMAE  += MAE.item()
    
    # Compute and save epoch loss (validation set)
    valEpochLoss = totalBatchValLoss / (j + 1)   
    valEpochMAE  = totalBatchValMAE / (j + 1)   
    
    return valEpochLoss, valEpochMAE



########################## Weighted mean squared error loss function ###########################
class weightedMSE:
    def __init__(self, MSEweightFactor):
        """
        Customised mean squared error loss function, weighted penalty based on groundtruth spherical equivalent refraction
        
        Args:        
              weightFactor (float) : Apply weight to loss for eyes with myopia < -3D (if set to 0 then standard/non-weighted MSE applied) 
        """
        
        self.MSEweightFactor = MSEweightFactor
        
        if self.MSEweightFactor != 0:
            print("Weighted MSE initialised")
        
    def __call__(self, output, target):  
        """
        Args:
              output (tensor float) : Model out (predicted spherical equivalent refraction).
              target (tensor float) : Groundtruth spherical equivalent refraction.
          
        Out:
              loss (tensor float)   : Weighted mean squared error (averaged across images in a batch)
        """
        
        device  = "cuda" if torch.cuda.is_available() else "cpu" 
        
        # Compute loss weight = MSEweightFactor if SER ≤ -5D or ≥ +3D)
        weights = torch.where((target < -8) | (target > +5), 
                              torch.tensor(6.0, dtype = torch.float32, device = device),              # 6 for SER < -8 or > +5
                              torch.where((target <= -5) | (target >= +3),    
                                          torch.tensor(3.0, dtype = torch.float32, device = device),  # 3 for -8 ≤ SER ≤ -5 or +3 ≤ SER ≤ +5
                                          torch.tensor(1.0, dtype = torch.float32, device = device)   # 1 for SER > -5 or SER < +3
                                         )
                             )                                    
        
        loss    = weights*((output - target)**2)
        loss    = torch.mean(loss)
        
        return loss

    
    
######################## Weighted random sampler for training/validation ########################
class WeightedSubsetRandomSampler(Sampler):
    def __init__(self, subsetIndices, classWeights, replacement=True):
        """
        Custom data sampler used to oversample training examples with high myopia
        
        Args:
            subsetIndices (list)  : Indices for the subset.
            classWeights (list)   : Weights for each sample (must match dataset size).
            replacement (bool)    : Whether sampling is with replacement.
        """
        self.subsetIndices = np.array(subsetIndices)
        self.classWeights  = np.array(classWeights)[self.subsetIndices]
        self.replacement   = replacement

    def __iter__(self):
        sampledIndices = np.random.choice(self.subsetIndices, 
                                          size    = len(self.subsetIndices), 
                                          replace = self.replacement, 
                                          p       = self.classWeights / self.classWeights.sum() )
        
        return iter(sampledIndices)

    def __len__(self):
        
        return len(self.subsetIndices)    
   

    
##################################### Mixup augmentation #######################################
def mixup(image, GToneHot, alpha):
    """
    Args:
          image (4D tensor)    : Expect to have [B,C,H,W] shape.
          GToneHot (2D tensor) : One-hot encoded label; expect to have [B,numClasses] shape 
          alpha (float)        : factor that controls the extent of mixup by influencing the 
                                 beta distribution from which the lambda value ("lam") is sampled. 
                                 lambda ranges from 0 to 1, where 0.5 indicates a 50:50 mixup between
                                 the first image and the second image. Larger alpha values are more 
                                 likely to yield lambda values that are closer to 0.5, giving rise to
                                 stronger regularisation. Smaller alpha values, on the other hand, are
                                 more likely to yield lambda values that are closer to either 0 or 1, 
                                 giving rise to little mixup between images.
    Out:
          newImage (4D tensor) : Batch of new (composite) images; expect to have [B,C,H,W] shape.
          newGToneHot (list)   : Contains labels in their original order, labels in their shuffled
                                 order and the randomly sampled lambda value.   
    """
    
    # Randomly shuffle the order of images and their 
    # corresponding labels in the current training batch
    indices          = torch.randperm(image.size(0))
    shuffledImage    = image[indices]
    shuffledGToneHot = GToneHot[indices]          
    
    # Randomly sample a lambda ("lam") value from 
    # the beta distribution defined by alpha
    lam = np.random.beta(alpha, alpha)
    
    # New (composite) images are created by mixing images 
    # in their original order with images in their
    # shuffled order, i.e. two images per composite image
    newImage = image * lam + shuffledImage * (1 - lam)
    
    # Labels and lambda
    newGToneHot = [GToneHot, shuffledGToneHot, lam]
    
    return newImage, newGToneHot



######################################### Mixup loss ###########################################
def mixupCriterion(criterion, predProbs, targets):
    """
    Args:
          criterion (function)   : torch.nn loss function.
          predProbs (2D tensor)  : Predicted probability for each myopic maculopathy 
                                   category; expect to have [B, numClasses] shape.
          targets (list)         : "newGToneHot" output by the "mixup" function above.
    Out:
          mixupLoss (1D tensor) : Losses with len(mixupLoss) = [B].
    """
    
    # Unpack targets
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    
    # Compute mixup loss
    mixupLoss = lam * criterion(predProbs, targets1) + (1 - lam) * criterion(predProbs, targets2)
    
    return mixupLoss
