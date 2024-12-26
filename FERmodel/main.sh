#!/bin/bash  
# Train script
# Fabian Yii 25 Nov 2024
# fabian.yii@ed.ac.uk

date

echo "############ Train (cross-validation) #############"
python crossVal.py --plotLosses --imageDir /opt/notebooks/images/ --numWorkers 16 --kFolds 5 --numEpochs 50 

echo "########## Internal inference: train/val set ##########"
python inference.py --imageDir /opt/notebooks/images/ --numWorkers 16 --dataFile /opt/notebooks/retinalRefractionGap/data/retinalRefractionGap_train_new.csv --testWhatDataset train

echo "############# Internal inference: UK Biobank unseen set ##############"
python inference.py --TTA --imageDir /opt/notebooks/images/ --numWorkers 16 --dataFile /opt/notebooks/retinalRefractionGap/data/retinalRefractionGap_test_new.csv --testWhatDataset test 

echo "############# External inference: Caledonian (external) dataset ##############"
python inferenceExternal.py --TTA 






