# Author :  Fabian Yii                                                                   #
# Email  :  fabian.yii@ed.ac.uk                                                          #
# Info   :  Assess external generalisability of the FER deep learning model to MMAC data #

setwd("..")


## Supplementary S8: MMAC SER prediction task 

# Read data from the train and validation sets, and merge them
trainPred <- read.csv(file.path("MMACtask3", "predictedTrain.csv"))
trainGT   <- read.csv(file.path("MMACtask3", "GT", "MMAC2023_SER_train.csv"))
train     <- merge(trainGT, trainPred, by = "image")
valPred   <- read.csv(file.path("MMACtask3", "predictedValidation.csv"))
valGT     <- read.csv(file.path("MMACtask3", "GT", "MMAC2023_SER_validation.csv"))
val       <- merge(valGT, valPred, by = "image")
pred      <- rbind(train, val)

# Correlation between SER and fundus equivalent refraction
cor.test(pred$SER, pred$predSER_TTA)

# Compute fundus refraction offset (FRO)
m               <- lm(predSER_TTA ~ SER, pred); tab_model(m)
pred$FRO        <- residuals(m)
pred$FROclass   <- ifelse(pred$FRO < 0 , "- offset", "+ offset")

# Plot FER vs SER
ggplot(pred, aes(x = SER, y = predSER_TTA)) +
  geom_point(alpha = 0.5, aes(col = FROclass)) +
  geom_smooth(method = "lm") + 
  labs(x = "Spherical equivalent refraction (D)", 
       y = "Fundus equivalent refraction (D)") +
  guides(fill = guide_legend(title = "FRO"),
         col  = guide_legend(title = "FRO")) +
  scale_x_continuous(limits = c(-10, 3), breaks = seq(3, -10, -1), labels =  seq(3, -10, -1)) +
  scale_y_continuous(limits = c(-10, 3), breaks = seq(3, -10, -1), labels =  seq(3, -10, -1)) +
  theme_blank()
ggsave(file.path("manuscript", "figures", "FERvsSER_MMAC.png"), width = 7, height = 5)











