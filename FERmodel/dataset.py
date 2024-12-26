"""
This script defines the custom dataset class compatible with UK Biobank

Author : Fabian Yii
Email  : fabian.yii@ed.ac.uk or fslyii@hotmail.com

2024 
"""

import os
import torch
import numpy as np
import cv2 as cv
from oct_converter.readers import FDS
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset
from utils import augmentation
join = os.path.join 


class DatasetGenerator(Dataset):
    def __init__(self, dataFrame, imageDir, useRAP, train, getOCT = False, fundusCroppedSize = (1400,1400), resizedDim = (512,512)):
        """
        Args:
              dataFrame (pd.DataFrame)         : Dataframe containing names of input images (column must be called "name").
              imageDir (str)                   : Full path to the directory where input images are saved (set to None if "useRAP" is True).
              useRAP (bool)                    : Set to True if using UK Biobank's Research Analysis Platform (False if running locally)
              train (bool)                     : Set to True for training set (False for validation or test set), as it 
                                                 will activate data augmentation.
              getOCT (bool)                    : Get OCT scans if True (default is False).                                   
              fundusCroppedSize (tuple of int) : Desired size of the centre crop for UK Biobank fundus image (default to
                                                 1400 by 1400 pixels to remove the black/empty border around the image).
              resizedDim (tuple of int)        : Desired input size of fundus images and OCT B-scans.
        """
        self.dataFrame   = dataFrame
        self.imageDir    = imageDir if not useRAP else None # Only need to "imageDir" if "useRAP" is False
        self.useRAP      = useRAP        
        self.train       = train
        self.getOCT      = getOCT
        self.crop        = CenterCrop(fundusCroppedSize)
        self.resizedDim  = resizedDim
        
        # Initialise regular augmentations (random rotation, horizontal flip & brightness/saturation jitter).
        self.augment = augmentation(angles = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])

    def __getitem__(self, index):
        """
        Out:
              imageName (str)       : Image names of length [B].
              fundus (4D tensor)    : Batch of fundus images of shape [B,C,H,W].
              foveaScan (4D tensor) : Batch of OCT horizontal macular line scans of shape [B,C,H,W], where C=3 (comprising 
                                      3 B-scans at the fovea). Only returned if "getOCT" is True.
              SER (float)           : Ground-truth spherical equivalent refraction.
        """
        ID        = self.dataFrame.id.iloc[index]
        imageName = self.dataFrame.name.iloc[index]
        SER       = self.dataFrame.SER.iloc[index]
        eye       = self.dataFrame.eye.iloc[index]
        
        if self.useRAP:
            # Get full path to image file (FDS format)
            FDSname       = imageName.replace("png", "fds")
            imagePath     = join(os.sep, 
                                 "mnt", 
                                 "project", 
                                 "Bulk", 
                                 "Retinal Optical Coherence Tomography", 
                                 "FDS (right)" if eye == "RE" else "FDS (left)", 
                                 str(imageName[0:2]), 
                                 FDSname.replace("21016", "21014") if eye == "RE" else FDSname.replace("21015", "21012") )            
            
            # Try opening the FDS file 
            try:
                fds       = FDS(imagePath)
                fundus    = fds.read_fundus_image()                        # Fundus image (in RGB by default)
                octVolume = fds.read_oct_volume() if self.getOCT else None # OCT volume with 128 horizontal b-scans 
            
            # Fall back to the first image if an error occurs due to whatever reason (e.g. corrupted file)
            except:
                ID        = self.dataFrame.id.iloc[0]
                imageName = self.dataFrame.name.iloc[0]
                SER       = self.dataFrame.SER.iloc[0]
                eye       = self.dataFrame.eye.iloc[0]
                FDSname   = imageName.replace("png", "fds")
                imagePath = join(os.sep, 
                                 "mnt", 
                                 "project", 
                                 "Bulk", 
                                 "Retinal Optical Coherence Tomography", 
                                 "FDS (right)" if eye == "RE" else "FDS (left)", 
                                 str(imageName[0:2]), 
                                 FDSname.replace("21016", "21014") if eye == "RE" else FDSname.replace("21015", "21012") )            
                fds       = FDS(imagePath)
                fundus    = fds.read_fundus_image()  
                octVolume = fds.read_oct_volume() if self.getOCT else None # OCT volume with 128 horizontal b-scans 
                print("Fall back to the first example because current file may be corrupted!")
            
            # Get fundus image
            fundus    = fundus.image.copy()    
            
            # Extract the 63th, 64th and 65th OCT scans (correspond to the foveal region)
            foveaScan = np.dstack((octVolume.volume[62], octVolume.volume[63], octVolume.volume[64])) if self.getOCT else None
      
        else:
            # Try opening the image file 
            try: 
                # Get full path to the fundus directory
                fundusDirPath  = join(self.imageDir, "fundus")
                fundus         = cv.cvtColor(cv.imread(join(fundusDirPath, imageName)), cv.COLOR_BGR2RGB)        
                
                # Get full path to the OCT directory
                if self.getOCT:
                    octDirPath = join(self.imageDir, "OCT", str(ID) + "_" + str(eye)) 
                    OCTSlice63 = cv.cvtColor(cv.imread(join(octDirPath, "slice_63.png")), cv.COLOR_BGR2GRAY)
                    OCTSlice64 = cv.cvtColor(cv.imread(join(octDirPath, "slice_64.png")), cv.COLOR_BGR2GRAY)
                    OCTSlice65 = cv.cvtColor(cv.imread(join(octDirPath, "slice_65.png")), cv.COLOR_BGR2GRAY)
                
            # Fall back to the first image if an error occurs due to whatever reason (e.g. corrupted file)
            except:
                ID             = self.dataFrame.id.iloc[0]
                imageName      = self.dataFrame.name.iloc[0]
                SER            = self.dataFrame.SER.iloc[0]
                eye            = self.dataFrame.eye.iloc[0]
                
                # Get full path to the fundus directory
                fundusDirPath  = join(self.imageDir, "fundus")
                fundus         = cv.cvtColor(cv.imread(join(fundusDirPath, imageName)), cv.COLOR_BGR2RGB)    
                
                # Get full path to the OCT directory
                if self.getOCT:
                    octDirPath = join(self.imageDir, "OCT", str(ID) + "_" + str(eye)) 
                    OCTSlice63 = cv.cvtColor(cv.imread(join(octDirPath, "slice_63.png")), cv.COLOR_BGR2GRAY)
                    OCTSlice64 = cv.cvtColor(cv.imread(join(octDirPath, "slice_64.png")), cv.COLOR_BGR2GRAY)
                    OCTSlice65 = cv.cvtColor(cv.imread(join(octDirPath, "slice_65.png")), cv.COLOR_BGR2GRAY)
                    foveaScan  = np.dstack((OCTSlice63, OCTSlice64, OCTSlice65))   
                    
                print("Fall back to the first example because current file may be corrupted!")
            
        # Centre crop fundus image to "croppedSize"
        fundus = self.crop(torch.as_tensor(fundus).moveaxis(2,0)).moveaxis(0,2).detach().numpy() 
        
        # Resize fundus image to resizedDim pixels
        fundus = cv.resize(fundus, self.resizedDim, interpolation = cv.INTER_AREA)
        
        # Normalise fundus image to [0-1] range and reshape into [C,H,W]  
        fundus = torch.as_tensor(fundus)
        fundus = fundus.moveaxis(2,0) / torch.max(fundus)  
        
        if self.getOCT:
            # Make sure foveaScan is of float data type
            foveaScan = foveaScan.astype(np.float32)
            
            # Each OCT scan is 650 (height) by 512 (width) pixels in the UK Biobank; remove the bottom 138 pixels as they aren't useful
            foveaScan = foveaScan[0:512, 0:512]
            
            # Resize the scan to resizedDim pixels
            foveaScan = cv.resize(foveaScan, self.resizedDim, interpolation = cv.INTER_AREA)
            
            # Normalise foveal OCT scan to [0-1] range and reshape into [C,H,W]  
            foveaScan = torch.as_tensor(foveaScan)
            foveaScan = foveaScan.moveaxis(2,0)/torch.max(foveaScan) 
        
        # Data augmentation (applies to training images only)
        if self.train:
            fundus    = self.augment(fundus, randomRotate = True)
            foveaScan = self.augment(foveaScan, randomRotate = False) if self.getOCT else None
        
        if self.getOCT:
            return imageName, SER, fundus, foveaScan
        else:
            return imageName, SER, fundus

    def __len__(self):
        return len(self.dataFrame.name)