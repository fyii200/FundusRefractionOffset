"""
This script contains custom helper functions

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.nn.functional import softmax
from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import roc_auc_score as AUC
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, functional, Lambda
join = os.path.join

################################################################################################
#################################### Regular augmentations #####################################
################################################################################################
class augmentation:
    def __init__(self, angles):
        """
        Args:
              angles (list): List of angles to be randomly sampled from.
        """
            
        self.angles = angles

    def __call__(self, inputImage):
        """
        Args:
              inputImage (3D or 4D tensor)  : Expect to have [...,C,H,W] shape.
        
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
        outputImage = cj(Hflip(functional.rotate(inputImage, angle)))
        
        return outputImage

    
################################################################################################
##################################### Mixup augmentation #######################################
################################################################################################
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
          newImage (4D tensor): Batch of new (composite) images; expect to have [B,C,H,W] shape.
          newGToneHot (list)  : Contains labels in their original order, labels in their shuffled
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


################################################################################################
######################################### Mixup loss ###########################################
################################################################################################
def mixupCriterion(criterion, output, targets):
    """
    Args:
          criterion (function)   : torch.nn loss function.
          output (2D tensor)     : Model output, one for each pathologic myopia category;
                                   expect to be of [B, numClasses] shape.
          targets (list)         : "newGToneHot" output by the "mixup" function above.
    Out:
          mixupLoss (1D tensor) : Losses with len(mixupLoss) = [B].
    """
    
    # Unpack targets
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    
    # Compute mixup loss
    mixupLoss = lam * criterion(output, targets1) + (1 - lam) * criterion(output, targets2)
    
    return mixupLoss


################################################################################################
########################## Train step (one step completes one epoch) ###########################
################################################################################################
def trainStep(trainDataLoader, model, mixupProb, mixupAlpha, augmentFn, lossFn, optimiser, LRscheduler, device):
    """
    Run one training step (i.e. one epoch)
    
    Args:
          trainDataLoader (torch.utils.data.DataLoader) : Initialised dataloader for the training set.
          model (torchvision.models)                    : Initialised regression model.
          mixupProb (float)                             : Probability of applying mixup augmentation (no mix-up if equal to 0).
          mixupAlpha (float)                            : Larger values result in greater mix-up between images (only relevant if mixupProb>0).
          augmentFn (function)                          : Initialised function for data augmentation.
          lossFn (torch.nn)                             : Initialised loss function.
          optimiser (torch.optim)                       : Initialised optimising algorithm.
          LRscheduler (torch.optim.lr_scheduler)        : Initialised learning rate scheduler.
          device (str)                                  : "mps", "cuda" or "cpu".
          
    Out:
          trainEpochLoss (tensor float)                 : Mean loss in a training epoch.
          trainEpochAUC (tensor float)                  : Mean area under the receiver operating characteristics curve in current training epoch.
    """
    # Train mode
    model.train()

    # Placeholder for batch losses summed across a training epoch
    metrics = defaultdict(list)
    totalTrainBatchLoss = 0.
  
    for i, (name, image, GToneHot, GTbinary) in enumerate(trainDataLoader):
        image    = image.to(device, torch.float)    
        GToneHot = GToneHot.to(device) 
        GTbinary = GTbinary.to(device) 

        ##################################################################
        #  Apply Mixup augmentation to the entire batch with a 0.5 prob. #
        ##################################################################
        p = np.random.rand()
        if p < mixupProb:
              image, GToneHot = mixup(image, GToneHot, alpha=mixupAlpha)
        ##################################################################
        
        # Apply regular augmentations. Note that this is applied to one image at 
        # a time as opposed to applying it similarly to the entire batch at once.
        image = Lambda(lambda image: torch.stack([augmentFn(x) for x in image]))(image)
        
        # Get model output and predicted probability of having pathologic myopia
        output         = model(image)
        predictedProb  = softmax(output, dim = 1)[:,1] 

        ##################################################################
        #    Apply Mixup criterion if mixup augmentation was applied.    #
        ##################################################################    
        if p < mixupProb:
              loss = mixupCriterion(lossFn, output, GToneHot)
        else:
              loss = lossFn(output, GToneHot)
        ##################################################################
            
        # Backprop
        loss.backward() 
        optimiser.step()  
        optimiser.zero_grad(set_to_none = True)

        # Save batch metrics
        totalTrainBatchLoss += loss.item() 
        [metrics["predictedProb"].append(i)  for i in predictedProb.cpu().detach().numpy() ]
        [metrics["GTbinary"].append(i)       for i in GTbinary.cpu().detach().numpy()      ]
       
    # Update learning rate
    LRscheduler.step()
      
    # Compute and save epoch metrics
    metrics          = pd.DataFrame(metrics)
    trainEpochLoss   = totalTrainBatchLoss / (i+1)
    trainEpochAUC    = AUC(metrics.GTbinary, metrics.predictedProb) if len(np.unique(metrics.GTbinary)) > 1 else print("Only one class present in 'GTbinary', skipping AUC calculation")
    return trainEpochLoss, trainEpochAUC


################################################################################################
######################## Validation step (one step completes one epoch) ########################
################################################################################################
def valStep(valDataLoader, model, lossFn, TTA, device):
    """
    Run one validation step (i.e. one epoch) — always preceded by a training step
    
    Args:
          valDataLoader (torch.utils.data.DataLoader) : Initialised dataloader for the validation set.
          model (torchvision.models)                  : Initialised regression model.
          lossFn (torch.nn)                           : Initialised loss function.
          TTA (bool)                                  : Test-time agmentations (horizontal flip & rotation) are applied if True.
          device (str)                                : "mps", "cuda" or "cpu".
          
    Out:
          valEpochLoss (tensor float)                 : Mean loss in current validation epoch.
          sensitivities (dict)                        : Mean sensitivities in current validation epoch using different thresholds.
          specificities (dict)                        : Mean specificities in current validation epoch using different thresholds.
          valEpochAUC (tensor float)                  : Mean area under the receiver operating characteristics curve in current validation epoch.
    """
    # Evaluation mode
    model.eval()

    # Placeholder for batch losses summed across a validation epoch
    metrics = defaultdict(list)
    totalValBatchLoss = 0.
    
    for i, (name, image, GToneHot, GTbinary) in enumerate(valDataLoader):
        image    = image.to(device, torch.float)    
        GToneHot = GToneHot.to(device) 
        GTbinary = GTbinary.to(device) 

        # Get model output from the original (non-augmented) image
        output   = model(image)
        
        # Predict model output (one for each class) from the original image and variants of the same image (if "TTA" is True)
        if TTA: 
                testHorFlip      = RandomHorizontalFlip(p = 1)
                outputHorFlipped = model(testHorFlip(image))         
                outputRotated1   = model(functional.rotate(image, -5))
                outputRotated2   = model(functional.rotate(image, 5))               
                # Take the average
                outputSummed   = (output + 
                                  outputHorFlipped + 
                                  outputRotated1   + 
                                  outputRotated2)                                  
                finalOutput     = outputSummed / 4
        else:
                finalOutput     = output
                
        # Turn model output into predicted probability of pathologic myopia
        predictedProb           = softmax(finalOutput, dim = 1)[:,1] 
        
        # Compute loss
        loss = lossFn(finalOutput, GToneHot)

        # Save batch metrics
        totalValBatchLoss += loss.item() 
        
        [metrics["GTbinary"].append(i)      for i in GTbinary.cpu().detach().numpy()     ]    
        [metrics["predictedProb"].append(i) for i in predictedProb.cpu().detach().numpy()]
   
    # Compute and save epoch metrics
    metrics         = pd.DataFrame(metrics)
    valEpochLoss    = totalValBatchLoss / (i+1)
    thresholds      = np.arange(0.02, 0.51, 0.01)
    sensitivities   = {}
    specificities   = {}
    
    for threshold in thresholds:
        predictedClass = (metrics["predictedProb"] > threshold).astype(int)
        sensitivity    = np.sum((metrics["GTbinary"] == 1) & (predictedClass == 1)) / np.sum(metrics["GTbinary"] == 1)
        specificity    = np.sum((metrics["GTbinary"] == 0) & (predictedClass == 0)) / np.sum(metrics["GTbinary"] == 0)
        sensitivities[f"valEpochSens{threshold:.2f}"] = sensitivity
        specificities[f"valEpochSpec{threshold:.2f}"] = specificity
        
    fpr, tpr, ROCthresholds = ROC(metrics["GTbinary"], metrics["predictedProb"])    
    valEpochAUC             = AUC(metrics.GTbinary, metrics.predictedProb) if len(np.unique(metrics.GTbinary)) > 1 else print("Only one class present in 'GTbinary', skipping AUC calculation")
    
    return valEpochLoss, sensitivities, specificities, valEpochAUC 

    
    









        
        