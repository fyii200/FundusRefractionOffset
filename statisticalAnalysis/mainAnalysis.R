# Author :  Fabian Yii                                                      #
# Email  :  fabian.yii@ed.ac.uk                                             #
# Info   :  Compute fundus refraction offset for the UK Biobank unseen set  #
#           and the Caledonian dataset, perform statistical analysis, and   #
#           create plots.                                                   #

rm(list=ls())
library(ggplot2)
library(lme4)
library(sjPlot)
library(car)
library(dplyr)
library(stringr)
library(forcats)
library(gridExtra)
library(scales)
library(viridis)
library(RColorBrewer)
library(grid)
setwd("..")



####################### UK Biobank ######################

## Read data and compute fundus refraction offset (FRO) ##

# UK Biobank cross-validation data
trainMetrics         <- read.csv(file.path("code", "FERmodel", "metrics", "CrossVal.csv"))
epochLosses          <- trainMetrics %>% group_by(epoch) %>%  summarise(epochLoss = mean(valEpochLoss)) 
bestEpoch            <- which(epochLosses$epochLoss == min(epochLosses$epochLoss)); print(paste("Best epoch:", bestEpoch))

# UK Biobank full participant data
d                    <- read.csv(file.path("data", "finalData.csv"))

# Create a binary indicator for OCT quality
upperQ               <- 0.95 # upper quantile
lowerQ               <- 0.05 # lower quantile
d$OCTgoodQualityBool <- d$OCT_quality_score                >= 45                                                              & 
                        d$OCT_ILM_indicator_baseline       < quantile(d$OCT_ILM_indicator_baseline, upperQ, na.rm=TRUE)       & 
                        d$OCT_ILM_indicator_baseline       > quantile(d$OCT_ILM_indicator_baseline, lowerQ, na.rm=TRUE)       & 
                        d$OCT_macula_centre_aline_baseline < quantile(d$OCT_macula_centre_aline_baseline, upperQ, na.rm=TRUE) & 
                        d$OCT_macula_centre_aline_baseline > quantile(d$OCT_macula_centre_aline_baseline, lowerQ, na.rm=TRUE) &
                        d$OCT_macula_centre_frame_baseline < quantile(d$OCT_macula_centre_frame_baseline, upperQ, na.rm=TRUE) & 
                        d$OCT_macula_centre_frame_baseline > quantile(d$OCT_macula_centre_frame_baseline, lowerQ, na.rm=TRUE) &
                        d$OCT_max_motion_delta_baseline    < quantile(d$OCT_max_motion_delta_baseline, upperQ, na.rm=TRUE)    &
                        d$OCT_max_motion_factor_baseline   < quantile(d$OCT_max_motion_factor_baseline, upperQ, na.rm=TRUE)   &
                        d$OCT_min_motion_corr_baseline     > quantile(d$OCT_min_motion_corr_baseline, lowerQ, na.rm=TRUE)     & 
                        d$OCT_valid_count_baseline         > quantile(d$OCT_valid_count_baseline, lowerQ, na.rm=TRUE)

# UK Biobank fundus equivalent refraction (FER) data
train                <- read.csv(file.path("output", "FERtrainPrediction.csv"))
train$predSER_TTA    <- NA
test                 <- read.csv(file.path("output", "FERtestPrediction.csv"))
train$type           <- "Train"
test$type            <- "Test"

# Merge participant and FER data
FERdata              <- rbind(train, test)
FERdata              <- merge(FERdata, d, by = "name" )
FERdata$type         <- factor(FERdata$type, c("Train", "Test"))

# UK Biobank MT distribution before OCT quality control
ggplot(subset(FERdata, type == "Test"), aes(x = overall_macular_thickness_baseline)) + 
  labs(x = "Overall macular thickness (µm)", y = "Frequency") +
  geom_histogram(bins = 200, fill = "red") + 
  theme_blank()
ggsave(file.path("manuscript", "figures", "supplementary", "MTdist.png"), width = 7, height = 5)

# Exclude eyes failing OCT quality control
excludeTestIds <- which(FERdata$type == "Test" & !FERdata$OCTgoodQualityBool)
FERdata        <- FERdata[-excludeTestIds,] 

# Eye-specific Pearson's correlation (FER vs SER)
for(eye in c("RE", "LE")){
  print(cor.test(subset(FERdata, type == "Train" & eye == eye)$trueSER, subset(FERdata, type == "Train" &  eye == eye)$predSER))
  print(cor.test(subset(FERdata, type == "Test" &  eye == eye)$trueSER, subset(FERdata, type == "Test" &  eye == eye)$predSER_TTA)) }

# Compute FRO for the UK Biobank unseen set
FERtest          <- subset(FERdata, type == "Test")
FERtestModel     <- lmer(predSER_TTA ~ trueSER + (1 | id), FERtest)
FERtest$FRO      <- residuals(FERtestModel)

## Associations of FRO with macular thickness (MT) ##

# Overall
tab_model(lmer(overall_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Central
tab_model(lmer(central_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Inner temporal
tab_model(lmer(innerTemporal_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Inner inferior
tab_model(lmer(innerInferior_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Inner nasal
tab_model(lmer(innerNasal_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Inner superior
tab_model(lmer(innerSuperior_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Outer temporal
tab_model(lmer(outerTemporal_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Outer inferior
tab_model(lmer(outerInferior_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Outer nasal
tab_model(lmer(outerNasal_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))

# Outer superior
tab_model(lmer(outerSuperior_macular_thickness_baseline ~ trueSER + FRO + age + sex + ethnicBinary + (1|id), FERtest))



############## Caledonian (external dataset) ############

## Read relevant data and compute FRO ##

# Read FER data 
pred    <- read.csv(file.path("output", "SCOOBI_FER.csv"))             

# Read groundtruth (SER) data
GT      <- read.csv(file.path("SCOOBI", "data", "SCOOBIdataLong.csv")) 

# Read choroid data
choroid <- read.csv(file.path("SCOOBI", "data", "choroid.csv"))        

# Merge dataframes
pred$ID <- NA
for(i in 1:nrow(pred)){
  splitName    <- str_split(pred$name[i], "_")[[1]]
  pred$ID[i]   <- splitName[1] }
pred           <- merge(GT, pred, by = "ID")
pred           <- merge(pred, choroid, all.x = TRUE, by ="ID")
pred           <- subset(pred, !is.na(SER) & cyclo == TRUE & eye == "RE")

# Binarise SER (myopia vs non-myopia) and bin axial length (quintiles)
pred$SERclass           <- ifelse(pred$SER <= -0.50, "Myopia", "Non-myopia")
pred$ALquantile         <- factor(as.numeric(cut_number(pred$meanAL, 5)))
levels(pred$ALquantile) <- c("20.7 - 22.8 mm", "22.9 - 23.4 mm", "23.5 - 23.9 mm", "24.0 - 24.4 mm", "24.5 - 27.3 mm")

# Compute overall MT
pred$overallMT    <- (pred$MT_central + pred$MT_IN + pred$MT_II + pred$MT_IT + pred$MT_IS + pred$MT_ON + pred$MT_OI + pred$MT_OT + pred$MT_OS) / 9

# Binarise ethnicity into "White" and "Non-white"
pred$ethnicBinary <- ifelse(pred$ethnic == "White" | pred$ethnic == "Caucasian", "White", "Non-white")

# Pearson's correlation between SER and FER
cor.test(pred$SER, pred$predSER_TTA)

# Compute FRO
m               <- lm(predSER_TTA ~ SER, pred); tab_model(m)
pred$FRO        <- residuals(m)
pred$FROclass   <- ifelse(pred$FRO < 0 , "—", "+")

## Associations of FRO with MT, CA & CVI (controlling for SER and then AL) ##

# overall MT
tab_model(lm(overallMT ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(overallMT ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Central MT
tab_model(lm(MT_central ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_central ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Inner temporal MT
tab_model(lm(MT_IT ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_IT ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Inner inferior MT
tab_model(lm(MT_II ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_II ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Inner nasal MT
tab_model(lm(MT_IN ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_IN ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Inner superior MT
tab_model(lm(MT_IS ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_IS ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Outer temporal MT
tab_model(lm(MT_OT ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_OT ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Outer inferior MT
tab_model(lm(MT_OI ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_OI ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Outer nasal MT
tab_model(lm(MT_ON ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_ON ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Outer superior MT
tab_model(lm(MT_OS ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(MT_OS ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Choroidal area (CA)
tab_model(lm(meanChoroidArea ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(meanChoroidArea ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))

# Choroidal vascularity index (CVI)
tab_model(lm(meanChoroidVascularity ~ SER + FRO + ageFundus + gender + ethnicBinary, pred))
tab_model(lm(meanChoroidVascularity ~ meanAL + FRO + ageFundus + gender + ethnicBinary, pred))



######################### Plots #########################

# Set-up: colour blindness friendly colour palette 
CB <- brewer.pal(11, "RdYlBu")

# Set-up: create new dataframe for plotting purposes
FERtest[FERtest$ethnicBinary == "non-White",]$ethnicBinary <- "Non-white"
FERtest$FROclass                                           <- ifelse(FERtest$FRO <= 0, "—", "+")
FERtest$SERclass                                           <- ifelse(FERtest$SER <= -0.50, "Myopia", "Non-myopia")
FERtest$ageGroup                                           <- "50-59y" 
FERtest[FERtest$age <= 49, ]$ageGroup                      <- "40-49y"
FERtest[FERtest$age >= 60, ]$ageGroup                      <- "60-70y"
plotData <- data.frame("data"       = c(rep("UK Biobank", nrow(FERtest)), rep("Caledonian", nrow(pred))), 
                       "MT"         = c(FERtest$overall_macular_thickness_baseline, pred$overallMT), 
                       "sex"        = c(FERtest$sex, pred$gender),
                       "ethnic"     = c(FERtest$ethnicBinary, pred$ethnicBinary), 
                       "ageGroup"   = c(FERtest$ageGroup, rep(" ", nrow(pred)) ), 
                       "ALquantile" = c(rep(NA, nrow(FERtest)), pred$ALquantile),
                       "FROclass"   = c(FERtest$FROclass, pred$FROclass), 
                       "SERclass"   = c(FERtest$SERclass, pred$SERclass))

# UK Biobank train set: FER vs SER
trainPlot <- ggplot(subset(FERdata, type == "Train"), aes(x = trueSER, y = predSER)) + 
             geom_point(col = "gray", alpha = 0.6) + 
             geom_smooth(method = "lm", col = "black") +
             labs(x = "", y = "", caption = "UK Biobank train set")  +
             theme_blank() +
             theme(legend.position  = "none",
                   axis.ticks       = element_line("black"),
                   axis.title.y     = element_text(size = 16),
                   plot.subtitle    = element_text(hjust = 1),
                   panel.background = element_blank(),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank()) 
ggsave(file.path("manuscript", "figures", "figure1_train.png"), plot = trainPlot, width = 6, height = 8)

# UK Biobank (internal) unseen set: FER vs SER
internalTestPlot <- ggplot(FERtest, aes(x = trueSER, y = predSER, col = FROclass)) + 
                    geom_point(alpha = 0.4) + 
                    scale_colour_manual(values = c(CB[3], CB[9])) +
                    geom_smooth(method = "lm", col = "black") +
                    labs(x = "", y = "", caption = "Internal (UK Biobank) unseen")  +
                    theme_blank() +
                    guides(col = guide_legend(title = "Fundus refraction offset", override.aes = list(size = 4))) +
                    theme(legend.position  = "top",
                          legend.text      = element_text(size  = 14),
                          legend.title     = element_text(size  = 14),
                          axis.ticks       = element_line("black"),
                          plot.subtitle    = element_text(hjust = 1),
                          panel.background = element_blank(),
                          panel.grid.major = element_blank(),
                          panel.grid.minor = element_blank()) 
ggsave(file.path("manuscript", "figures", "figure1_internalUnseen.png"), plot = internalTestPlot, width = 6, height = 4)

# Caledonian (external) dataset: FER vs SER
externalTestPlot <- ggplot(pred, aes(x = SER, y = predSER_TTA, col = FROclass)) + 
                    geom_point(alpha = 0.4) + 
                    scale_colour_manual(values = c(CB[3], CB[9] )) +
                    scale_x_continuous(limits = c(-7, 8), breaks = seq(-7, 8, 3), labels =  seq(-7, 8, 3)) +
                    scale_y_continuous(limits = c(-4, 8.4), breaks = seq(-4, 8, 3), labels = seq(-4, 8, 3)) +
                    geom_smooth(method = "lm", col = "black", se = FALSE) +
                    labs(x = "", y = "", caption = "External (Caledonian) unseen")  +
                    theme_blank() +
                    theme(legend.position  = "none",
                          axis.ticks       = element_line("black"),
                          plot.subtitle    = element_text(hjust = 1),
                          panel.background = element_blank(),
                          panel.grid.major = element_blank(),
                          panel.grid.minor = element_blank()) 
ggsave(file.path("manuscript", "figures", "figure1_external.png"), plot = externalTestPlot, width = 6, height = 4)

# UK Biobank unseen set: MT distribution (post-OCT quality control)
ggplot(subset(FERdata, type == "Test"), aes(x = overall_macular_thickness_baseline)) + 
  labs(x = "Overall macular thickness (µm)", y = "Frequency") +
  geom_histogram(bins = 200, fill = "red") + 
  theme_blank()
ggsave(file.path("manuscript", "figures", "supplementary", "MTdistQC.png"), width  = 7, height = 5)

# UK Biobank unseen set: FRO distribution
ggplot(FERtest, aes(x = FRO)) + 
  geom_histogram(col = "gray", fill = "gray88") + 
  labs(x = "\nFundus refraction offset (D)", y = "Frequency\n", subtitle = "UK Biobank") +
  theme_blank() + 
  theme(axis.line = element_blank())
ggsave(file.path("manuscript", "figures", "supplementary", "FROdistUKbiobank.png"), width = 4, height = 4)

# Caledonian dataset: MT distribution
ggplot(pred, aes(x = overallMT)) + 
  labs(x = "Overall macular thickness (µm)", y = "Frequency") +
  geom_histogram(bins = 20, col = "transparent") + 
  scale_x_continuous(breaks = seq(250, 315, 5), labels = seq(250, 315, 5)) +
  theme_blank()
ggsave(file.path("manuscript", "figures", "supplementary", "MTdistGCU.png"), width = 7, height = 5, dpi = 300)

# Caledonian dataset: CA distribution
ggplot(pred, aes(x = meanChoroidArea)) + 
  labs(x = "Choroidal area (mm²)", y = "Frequency") +
  geom_histogram(bins = 20, col = "transparent") + 
  scale_x_continuous(breaks = seq(0.2, 2, 0.2), labels = format(seq(0.2, 2, 0.2),nsmall=2) ) +
  theme_blank()
ggsave(file.path("manuscript", "figures", "supplementary", "CAdistGCU.png"), width = 7, height = 5, dpi = 300)

# Caledonian dataset: CVI distribution
ggplot(pred, aes(x = meanChoroidVascularity)) + 
  labs(x = "Choroidal vascularity index", y = "Frequency") +
  geom_histogram(bins = 20, col = "transparent") + 
  scale_x_continuous(breaks = seq(0.3, 0.6, 0.05), labels = seq(0.3, 0.6, 0.05)) +
  theme_blank()
ggsave(file.path("manuscript", "figures", "supplementary", "CVIdistGCU.png"), width = 7, height = 5, dpi = 300)

# Caledonian dataset: FRO distribution
ggplot(pred, aes(x = FRO)) + 
  geom_histogram(col = "gray", fill = "gray88") + 
  labs(x = "\nFundus refraction offset (D)", y = "", subtitle = "Caledonian") +
  theme_blank() 
ggsave(file.path("manuscript", "figures", "supplementary", "FROdistGCU.png"), width = 4, height = 4)

# UK Biobank unseen set and Caledonian dataset: MT vs FRO 
# Stratified by refractive error (MTplotA) and AL (MTplotB)
MTplotA    <- ggplot(plotData, aes(x = SERclass, y = MT, fill = FROclass, col = FROclass)) +
              geom_boxplot(alpha = 0.4) +
              labs(x = "", y = "") +
              scale_y_continuous(limits=c(215, 356), breaks = seq(215,356,20), labels = seq(215,356,20)) +
              scale_color_manual(values = c(CB[3], CB[9])) +
              scale_fill_manual(values = c(CB[3], CB[9])) +
              facet_grid( ~ factor(data, c("UK Biobank", "Caledonian")) + ageGroup) +
              theme_blank() +
              theme(legend.position = "none", 
                    axis.ticks.y    = element_line(color = "black"), 
                    strip.text      = element_text(size = 15), 
                    axis.text       = element_text(size = 11)) 
MTplotB    <- ggplot(pred, aes(x = ALquantile, y = overallMT, fill = FROclass, col = FROclass)) + 
              labs(x = "", y = "") +
              scale_y_continuous(limits=c(250, 312), breaks = seq(250,315,15), labels = seq(250,315,15)) +
              scale_color_manual(values = c(CB[3], CB[9])) +
              scale_fill_manual(values = c(CB[3], CB[9])) +
              guides(fill = guide_legend(title = "Fundus refraction offset"), col  = guide_legend(title = "Fundus refraction offset")) +
              geom_boxplot(alpha = 0.4) +
              theme_blank() +
              theme(legend.position = "top", 
                    legend.text     = element_text(size  = 14),
                    legend.title    = element_text(size  = 12),
                    legend.key.size = unit(1, "cm"),
                    axis.ticks.y    = element_line(color = "black"), 
                    axis.text       = element_text(size  = 11)) 
MTcombined <- grid.arrange(MTplotA, MTplotB, ncol = 1, nrow = 2, left = textGrob("Overall macular thickness (µm)", rot = 90, gp = gpar(fontsize = 20)))
ggsave(file.path("manuscript", "figures", "figure2pre.png"), plot = MTcombined, width = 8.5, height = 12)

# UK Biobank unseen set and Caledonian dataset: CVI vs FRO 
# Stratified by refractive error (CVIplotA) and AL (CVIplotB)
CVIplotA    <- ggplot(pred) + 
               theme_blank() + 
               labs(x = "", y = "") +
               scale_y_continuous(limits=c(0.35, 0.60), breaks = seq(0.35, 0.60, 0.05), labels = label_number(accuracy = 0.01)) +
               scale_color_manual(values = c(CB[3], CB[9])) +
               scale_fill_manual(values = c(CB[3], CB[9])) +
               geom_boxplot(aes(x = SERclass, y = meanChoroidVascularity, fill = FROclass, col = FROclass), alpha=0.4) + 
               theme(legend.position = "none", axis.ticks.y = element_line(color = "black"))
CVIplotB    <- ggplot(pred, aes(x = ALquantile, y = meanChoroidVascularity, fill = FROclass, col = FROclass)) + 
               theme_blank() + 
               geom_boxplot(alpha = 0.4) + 
               guides(fill = guide_legend(title = "Fundus\nrefraction\noffset"), col  = guide_legend(title = "Fundus\nrefraction\noffset")) +
               scale_y_continuous(limits=c(0.35, 0.60)) +
               scale_color_manual(values = c(CB[3], CB[9])) +
               scale_fill_manual(values = c(CB[3], CB[9])) +
               labs(x = "", y = "") +
               geom_smooth(method = "lm", alpha = 0.1) +
               theme(axis.text.y = element_blank(), legend.position = "left")
CVIcombined <- grid.arrange(CVIplotA, CVIplotB, ncol = 2, nrow = 1, widths = c(0.45, 0.55), left = textGrob("Choroidal vascularity index", rot = 90, gp = gpar(fontsize = 16)))
ggsave(file.path("manuscript", "figures", "figure3pre.png"), plot = CVIcombined, width = 14, height = 7)

