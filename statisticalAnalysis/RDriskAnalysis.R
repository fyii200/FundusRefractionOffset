# Author :  Fabian Yii                                                      #
# Email  :  fabian.yii@ed.ac.uk                                             #
# Info   :  Association of baseline fundus refraction offset with 12-year   #
#           risk of retinal detachment or breaks                            #

rm(list=ls())
library(ggplot2)
library(gridExtra)
library(lme4)
library(sjPlot)
library(car)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(survival)
library(survminer)
# setwd("..")

options(scipen = 10000)




#######################################################################
##    Initial setup: read, preprocess and merge relevant datasets    ##
#######################################################################

# Read UK Biobank participant data 
d                    <- read.csv(file.path("data", "finalData.csv"))

# Read fundus equivalent refraction (FER) data
FERtest              <- read.csv(file.path("output", "FERtestPrediction.csv")) # unseen set

# Read first disease occurrence data 
firstOccur           <- read.csv(file.path("data", "baseline_withFundus_firstOccurences.csv")) 
firstOccur           <- firstOccur %>% select(Participant.ID, 
                                              Month.of.birth,
                                              Date.of.attending.assessment.centre...Instance.0,
                                              Date.of.death...Instance.0,
                                              Underlying..primary..cause.of.death..ICD10...Instance.0,
                                              Date.lost.to.follow.up,
                                              Reason.lost.to.follow.up,
                                              Age.started.wearing.glasses.or.contact.lenses...Instance.0,
                                              Age.when.loss.of.vision.due.to.injury.or.trauma.diagnosed...Instance.0,
                                              Date.H33.first.reported..retinal.detachments.and.breaks.,
                                              Source.of.report.of.H33..retinal.detachments.and.breaks.,
                                              Date.H36.first.reported..retinal.disorders.in.diseases.classified.elsewhere.,
                                              Source.of.report.of.H36..retinal.disorders.in.diseases.classified.elsewhere.)

# Rename the columns in first occurrence dataset
names(firstOccur)    <- c('id', 
                          'MOB', 
                          'visitDate', 
                          'dateDeath', 
                          'primaryReasonDeath', 
                          'dateLossFU', 
                          'reasonLossFU', 
                          'ageFirstGlasses', 
                          'ageVisionLossDueToInjuryOrTrauma', 
                          'firstDateRetinalDetachmentBreaks', 
                          'sourceRetinalDetachmentBreaks', 
                          'firstDateH36', 
                          'sourceH36')

# Ensure all date variables are correctly coded as datetime
cvtDate              <- function(x) format(as.Date(x, '%d/%m/%Y'))
firstOccur           <- firstOccur %>% mutate(visitDate                        = cvtDate(visitDate),
                                              dateDeath                        = cvtDate(dateDeath),
                                              dateLossFU                       = cvtDate(dateLossFU),
                                              firstDateRetinalDetachmentBreaks = cvtDate(firstDateRetinalDetachmentBreaks),
                                              firstDateH36                     = cvtDate(firstDateH36))

# Merge all datasets
FERtest              <- merge(d, FERtest, by = 'name')
FERtest              <- merge(FERtest, firstOccur, by = 'id', all.x = T)

# Read hospital operations data (coded based on OPCS-4); in long format where each operation has its own row
operations           <- read.csv(file.path('data', 'operations.csv'))
operations           <- operations %>% rename(id            = Participant.ID.participant...eid.,
                                              OPCS4         = Operative.procedures...OPCS4,
                                              operationDate = Date.of.operation) %>% 
                                       mutate(operationDate = cvtDate(operationDate))

# Extract subset of the operations dataset to include only participants included in this study 
# 7204 unique participants have at least one entry in the operations dataset
operations           <- operations[operations$id %in% FERtest$id, ]
length(unique(operations$id))

# Get first date cataract or lens extraction operation for each patient in FERtest
length(unique(subset(operations, substring(OPCS4, 1, 3) == 'C71')$id)) # 'C71' corresponds to 'Extracapsular extraction of lens' (573 patients)
length(unique(subset(operations, substring(OPCS4, 1, 3) == 'C72')$id)) # 'C72' corresponds to 'Intracapsular extraction of lens' (0 patients)
length(unique(subset(operations, substring(OPCS4, 1, 3) == 'C74')$id)) # 'C74' corresponds to 'Other extraction of lens' (2 patients)

# Extract subset with cataract surgery entries
catSurgery           <- subset(operations, substring(OPCS4, 1, 3) == 'C71' | substring(OPCS4, 1, 3) == 'C72' | substring(OPCS4, 1, 3) == 'C74')

# Compute number of surgeries per patient and difference in years between two adjacent surgeries if a patient had had more than 1 entry
catSurgery           <- catSurgery %>% group_by(id) %>% arrange(operationDate) %>% mutate(diffYear    = as.numeric(difftime(lead(operationDate), operationDate, units = 'days')/365.25),
                                                                                          nCatSurgery = length(operationDate),
                                                                                          TP          = 1:n())

# Convert cataract surgery dataset from long to wide format
catSurgery           <- catSurgery %>% ungroup(id) %>% pivot_wider(id_cols      = id, 
                                                                   names_from   = TP, 
                                                                   names_prefix = 'catSurgery', 
                                                                   values_from  = operationDate)

# Merge cataract surgery dataframe with the main dataframe
FERtest              <- merge(FERtest, catSurgery, by = 'id', all = T)

# Read hospital admissions diagnoses data (based on ICD-10)
diag                 <- read.csv(file.path('data', 'hospitalDiagnoses.csv'))
names(diag)[c(2, 9)] <- c('id', 'diag')

# Extract information on retinal detachment/break subcategory for each case
IDsWithRDFirstOccr   <- unique(subset(FERtest, !is.na(firstDateRetinalDetachmentBreaks))$id)
FERtest$RDtype       <- NA
for(i in IDsWithRDFirstOccr){
  RDbool <- substring(subset(diag, id == i)$diag, 1, 3) == 'H33'
  if(sum(RDbool) > 0){
    RDtype                            <- unique(subset(diag, id == i)$diag[RDbool])
    FERtest[FERtest$id == i, ]$RDtype <- ifelse(length(RDtype) == 1, RDtype, paste(RDtype)) } }

# Check how many participants had both eye data available (5130 with one eye only, 4190 both eyes)
FERtest              <- FERtest %>% group_by(id) %>% mutate(nEyes = n())
table(subset(FERtest, !duplicated(id))$nEyes)

# Average SER and FER across eyes if data from both eyes are available
FERtest              <- FERtest %>% mutate(indSER         = ifelse(n() > 1, (SER[1] + SER[2]) / 2, SER),
                                           indPredSER_TTA = ifelse(n() > 1, (predSER_TTA[1] + predSER_TTA[2]) / 2, predSER_TTA))

# Check inter-eye correlation for SER and FER
bothEyesWide         <- FERtest %>% pivot_wider(id_cols     = id, 
                                                names_from  = eye, 
                                                values_from = c(SER, predSER_TTA)) %>% drop_na(SER_RE, SER_LE)

# SER (pearson's correlation and plot LE vs RE)
cor.test(bothEyesWide$SER_RE, bothEyesWide$SER_LE)
ggplot(bothEyesWide, aes(x = SER_RE, y = SER_LE)) + 
      theme_minimal()                             +
      geom_point(alpha = 0.3)                     + 
      geom_smooth(method = 'lm')                  + 
      labs(x      = 'RE SER (D)', 
           y      = 'LE  SER (D)', 
          title   = 'N = 4190', 
          caption = 'SER, spherical equivalent refraction') 

# FER (pearson's correlation and plot LE vs RE)
cor.test(bothEyesWide$predSER_TTA_RE, bothEyesWide$predSER_TTA_LE)
ggplot(bothEyesWide, aes(x = predSER_TTA_RE, y = predSER_TTA_LE)) + 
       theme_minimal()                                            +
       geom_point(alpha = 0.3)                                    + 
       geom_smooth(method = 'lm')                                 + 
       labs(x       = 'RE FER (D)', 
            y       = 'LE  FER (D)', 
            title   = 'N = 4190', 
            caption = 'FER, fundus equivalent refraction') 




#######################################################################
##     Compute survival time and fundus refraction offset (FRO)      ##
#######################################################################

# Right-censoring date
lastFUdate                              <- as.Date('2022-05-31') 

# If visit (baseline assessment) date is missing, infer by summing year of birth and age at the baseline visit
missingVisitDate                        <- which(is.na(FERtest$visitDate))
if(length(missingVisitDate) > 0){
  FERtest[missingVisitDate, ]$visitDate <- format(as.Date(paste0(FERtest$YOB[missingVisitDate] + FERtest$age[missingVisitDate], '-01-01')), '%Y-%m-%d') }

# Create new columns in preparation for survival analysis 
# 'RD' indicates if a participant has a history of any subcatergory of RD/breaks during follow-up
# 'RDcumYear' is the time from baseline to the onset of any subcategory of RD/breaks or right-censoring date 
# 'rhegmaRD' indicates if a participant has a history of rhegmatogenous RD/breaks during follow-up
# 'rhegmaRDcumYear' is the time from baseline to the onset of rhegmatogenous RD/breaks or right-censoring date 
# 'timeToCataractOp' is the time from baseline to cataract surgery
# 'cataractOpDuring' indicates if a participant has undergone cataract surgery during follow-up and prior to disease onset or the right-censoring date
# 'ageTrauma' is the self-reported age when a participant had a self-reported trauma/injury 'resulting in a loss of vision'
# 'previousTrauma' is True if 'ageTrauma' is present and smaller than baseline age
FERtest                                 <- FERtest %>% mutate(RD                = ifelse(is.na(firstDateRetinalDetachmentBreaks), F, T),
                                                              RDcumYear         = ifelse(RD == T, difftime(firstDateRetinalDetachmentBreaks, visitDate)/365.25,
                                                                                         ifelse(!is.na(dateDeath), difftime(dateDeath, visitDate)/365.25,
                                                                                                ifelse(!is.na(dateLossFU), difftime(dateLossFU, visitDate)/365.25, difftime(lastFUdate, visitDate)/365.25))),
                                                              rhegmaRD          = ifelse((RDtype == 'H33.0 Retinal detachment with retinal break' | RDtype == 'H33.3 Retinal breaks without detachment') & !is.na(RDtype), T, F),
                                                              rhegmaRDcumYear   = ifelse(rhegmaRD == T, difftime(firstDateRetinalDetachmentBreaks, visitDate)/365.25,
                                                                                         ifelse(!is.na(dateDeath), difftime(dateDeath, visitDate)/365.25,
                                                                                                ifelse(!is.na(dateLossFU), difftime(dateLossFU, visitDate)/365.25, difftime(lastFUdate, visitDate)/365.25))),
                                                              timeToCataractOp  = difftime(catSurgery1, visitDate, units = 'days')/365.25,
                                                              cataractOpDuring  = ifelse(timeToCataractOp < RDcumYear & !is.na(catSurgery1), T, F),
                                                              ageTrauma         = ifelse(ageVisionLossDueToInjuryOrTrauma != '' & !is.na(ageVisionLossDueToInjuryOrTrauma) & ageVisionLossDueToInjuryOrTrauma != 'Do not know', as.numeric(ageVisionLossDueToInjuryOrTrauma), NA),
                                                              previousTrauma    = ifelse(!is.na(ageTrauma) & ageTrauma <= age, T, F))

# Remove duplicated IDs to keep only one row per participant, as event is defined 
# at the individual level and all eye-specific variables have been averaged
RDdata                                  <- subset(FERtest, !duplicated(id))

# Compute FRO at the individual level (using individual-level SER and FER)
FERmodel                                <- lm(indPredSER_TTA ~ indSER, RDdata)
RDdata$indFRO                           <- residuals(FERmodel)

# Create a new column storing categorised FRO (for visualisation purposes; Kaplan-Meier survival plot)
RDdata                                  <- RDdata %>% ungroup(id) %>% mutate(FROgroup = ifelse(indFRO < -1, '< -1.00', ifelse(indFRO < -0.5, '-1.00 to -0.49', ifelse(indFRO < 0, '-0.50 to -0.01', ifelse(indFRO < 0.5, '0 to +0.49', ifelse(indFRO <= 1, '+0.50 to +1.00', '> +1.00'))))),
                                                                             FROgroup = factor(FROgroup, c('> +1.00', '+0.50 to +1.00', '0 to +0.49', '-0.50 to -0.01', '-1.00 to -0.49', '< -1.00')))




#######################################################################
##      Descriptive statistics and Kaplan-Meier survival plot        ##
#######################################################################

# Summary stats for time to RD/breaks
quantile(subset(RDdata, RD == T)$RDcumYear)

# Number of participants with newly onset disease (n=64)
length(unique(RDdata[RDdata$RD == T, ]$id))

# 512 out of 9320 participants with a history of cataract surgery during follow-up
length(unique(subset(RDdata, cataractOpDuring == T)$id))

# Median year from baseline visit to cataract/lens extraction surgery
quantile(subset(RDdata, cataractOpDuring == T)$timeToCataractOp)

# 44 participants with a self-reported history of ocular trauma/injury
length(unique(subset(RDdata, previousTrauma == T)$id))

# Linkage sources of RD/breaks
# Of the 64 newly onset cases, majority (62) sourced from hospital data
table(subset(RDdata, !duplicated(id) & RD == T)$sourceRetinalDetachmentBreaks)

# Plot Kaplan-Meier survival curve stratified by baseline FRO (any RD/breaks as event)
allRD_KMfit        <- survfit(Surv(RDcumYear, RD) ~ FROgroup, RDdata)
allRD_KMplot       <- ggsurvplot(allRD_KMfit,
                                 ylim         = c(0.98, 1), 
                                 xlim         = c(0, 13.5),
                                 break.x.by   = 4,
                                 size         = 0.6,
                                 add.all      = T,
                                 censor       = F,
                                 xlab         = 'Years elapsed',
                                 ylab         = 'Survival probability\n',
                                 subtitle     = 'Any retinal detachment or breaks\n',
                                 legend       = 'right',
                                 legend.title = 'FRO (D)',
                                 legend.labs  = c('Overall', levels(RDdata$FROgroup)),
                                 ggtheme      = theme_minimal() + theme(panel.grid.major   = element_blank(), 
                                                                        panel.grid.minor   = element_blank(),
                                                                        plot.subtitle      = element_text(hjust = 0.5), 
                                                                        axis.title.x       = element_text(face = 'bold', hjust = 1),
                                                                        axis.title.y       = element_text(face = 'bold', hjust = 1)), 
                                 palette       = c('gray', brewer.pal(11, 'RdYlBu')[c(11, 9, 7, 4, 3, 1)]),
                                 risk.table    = T,
                                 tables.height = 0.2,
                                 fontsize      = 2.5,
                                 tables.y.text = F,
                                 tables.theme  = theme_cleantable() + theme(plot.title = element_text(size = 9, face = 'italic'))) 
allRD_KMplot$plot  <- allRD_KMplot$plot + guides(col = guide_legend(override.aes = list(linewidth = 2)))
pdf(file.path('manuscript2', 'figures', 'allRDkm.pdf'), width = 7.5, height = 6) 
print(allRD_KMplot)
dev.off()

# Plot Kaplan-Meier survival curve stratified by baseline FRO (rhegmatogenous RD/breaks as event)
rhegRD_KMfit       <- survfit(Surv(rhegmaRDcumYear, rhegmaRD) ~ FROgroup, RDdata)
rhegRD_KMplot      <- ggsurvplot(rhegRD_KMfit,
                                 ylim         = c(0.98, 1), 
                                 xlim         = c(0, 13.5),
                                 break.x.by   = 4,                             
                                 size         = 0.6,
                                 add.all      = T,
                                 censor       = F,
                                 xlab         = 'Years elapsed',
                                 ylab         = '\n',
                                 subtitle     = 'Rhegmatogenous retinal detachment or breaks\n',
                                 legend       = 'none',
                                 legend.title = '',
                                 legend.labs  = c('Overall', levels(RDdata$FROgroup)),
                                 ggtheme      = theme_minimal() + theme(panel.grid.major = element_blank(), 
                                                                        panel.grid.minor = element_blank(),
                                                                        plot.subtitle    = element_text(hjust = 0.5), 
                                                                        axis.title.x     = element_text(face = 'bold', hjust = 1),
                                                                        axis.title.y     = element_text(face = 'bold', hjust = 1)),
                                 palette       = c('gray', brewer.pal(11, 'RdYlBu')[c(11, 9, 7, 4, 3, 1)]),
                                 risk.table    = T,
                                 tables.height = 0.2,
                                 fontsize      = 2.5,
                                 tables.y.text = F,
                                 tables.theme  = theme_cleantable() + theme(plot.title = element_text(size = 9, face = 'italic')))                              
rhegRD_KMplot$plot <- rhegRD_KMplot$plot + guides(col = guide_legend(override.aes = list(linewidth = 2))) + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
pdf(file.path('manuscript2', 'figures', 'rhegmaRDkm.pdf'), width = 5.3, height = 6) 
print(grid.arrange(rhegRD_KMplot$plot, rhegRD_KMplot$table, ncol = 1, heights = c(2, 0.5)))
dev.off()




#######################################################################
##    Main analysis: univariable and multivariable Cox regression    ##
#######################################################################

## Any RD and retinal breaks as outcome
# Univariable
tab_model(coxph(Surv(RDcumYear, RD) ~ indSER, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ indFRO, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ age, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ sex, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ townsend, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ ethnicBinary, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ cataractOpDuring, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ diabetes, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ hypertension, data = RDdata))
tab_model(coxph(Surv(RDcumYear, RD) ~ BMI, data = RDdata))
# Multivariable
coxRD <- coxph(Surv(RDcumYear, RD) ~ indSER + indFRO + age + sex + cataractOpDuring, data = RDdata)
summary(coxRD); tab_model(coxRD)
# Diagnostic plots
plot(cox.zph(coxRD), 
     hr   = T, 
     bty  = 'n', 
     cex  = 0.5, 
     xlab = 'Years elapsed', 
     ylab = c('HR for SER', 'HR for FRO', 'HR for age', 'HR for sex', 'HR for cataract surgery during FU'))

## Rhegmatogenous RD and retinal breaks as outcome
# Univariable
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ indSER, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ indFRO, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ age, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ sex, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ townsend, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ ethnicBinary, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ cataractOpDuring, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ diabetes, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ hypertension, data = RDdata))
tab_model(coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ BMI, data = RDdata))
# Multivariable
coxRhegmaRD <- coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ indSER + indFRO + age + sex + cataractOpDuring, data = RDdata)
summary(coxRhegmaRD); tab_model(coxRhegmaRD)
# Diagnostic plots
plot(cox.zph(coxRhegmaRD), 
     hr   = T, 
     bty  = 'n', 
     cex  = 0.5, 
     xlab = 'Years elapsed', 
     ylab = c('HR for SER', 'HR for FRO', 'HR for age', 'HR for sex', 'HR for cataract surgery during FU'))




#######################################################################
##  Subgroup analysis: univariable and multivariable Cox regression  ##
#######################################################################

# Extract subset with macular thickness (MT) data
FERtestSub           <- subset(FERtest, !is.na(overall_macular_thickness_baseline))

# Create a binary indicator for OCT quality
upperQ               <- 0.95 # upper quantile
lowerQ               <- 0.05 # lower quantile
d$OCTgoodQualityBool <- d$OCT_quality_score                >= 45                                        & 
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

# Include scans passing quality control
includeFundi         <- unique(subset(d, OCTgoodQualityBool & !is.na(overall_macular_thickness_baseline))$name)
FERtestSub           <- FERtestSub %>% filter(name %in% includeFundi)

# Average MT across eyes if data from both eyes are available
FERtestSub           <- FERtestSub %>% group_by(id) %>% mutate(nEyes          = n(),
                                                               indSER         = ifelse(n() > 1, (SER[1] + SER[2]) / 2, SER),
                                                               indPredSER_TTA = ifelse(n() > 1, (predSER_TTA[1] + predSER_TTA[2]) / 2, predSER_TTA),
                                                               meanOverallMT  = ifelse(n() > 1, (overall_macular_thickness_baseline[1]+overall_macular_thickness_baseline[2])/2, overall_macular_thickness_baseline))
RDdataSub            <- subset(FERtestSub, !duplicated(id))

# Check inter-eye correlation for MT 
REoverallMT          <- subset(FERtestSub, nEyes == 2 & eye == 'RE')$overall_macular_thickness_baseline
LEoverallMT          <- subset(FERtestSub, nEyes == 2 & eye == 'LE')$overall_macular_thickness_baseline
cor.test(REoverallMT, LEoverallMT)

# 2397 participants with data from both eyes
length(REoverallMT)

# Plot LE MT vs RE MT
ggplot(data.frame(RE = REoverallMT, LE = LEoverallMT), aes(x = REoverallMT, y = LEoverallMT)) + 
  theme_minimal()            +
  geom_point(alpha = 0.3)    + 
  geom_smooth(method = 'lm') + 
  labs(x       = 'RE MT (µm)', 
       y       = 'LE  MT (µm)', 
       title   = 'N = 2397',
       caption = 'MT, baseline ETDRS macular thickness') 

# Compute FRO
FERmodel             <- lm(indPredSER_TTA ~ indSER, RDdataSub)
RDdataSub$indFRO     <- residuals(FERmodel) 

# Fit cox regression model (all RD & breaks)
coxRD                <- coxph(Surv(RDcumYear, RD) ~ indSER + indFRO + age + sex + cataractOpDuring + meanOverallMT, data = RDdataSub)
summary(coxRD); tab_model(coxRD)
# Diagnostic plots
plot(cox.zph(coxRD), 
     hr   = T, 
     bty  = 'n', 
     cex  = 0.5, 
     xlab = 'Years elapsed', 
     ylab = c('HR for SER', 'HR for FRO', 'HR for age', 'HR for sex', 'HR for cataract surgery during FU', 'HR for MT'))

# Fit cox regression model (rhegma RD & breaks only)
coxRhegmaRD          <- coxph(Surv(rhegmaRDcumYear, rhegmaRD) ~ indSER + indFRO + age + sex + cataractOpDuring + meanOverallMT, data = RDdataSub)
summary(coxRhegmaRD); tab_model(coxRhegmaRD)
# Diagnostic plots
plot(cox.zph(coxRhegmaRD), 
     hr   = T, 
     bty  = 'n', 
     cex  = 0.5, 
     xlab = 'Years elapsed', 
     ylab = c('HR for SER', 'HR for FRO', 'HR for age', 'HR for sex', 'HR for cataract surgery during FU', 'HR for MT'))


















