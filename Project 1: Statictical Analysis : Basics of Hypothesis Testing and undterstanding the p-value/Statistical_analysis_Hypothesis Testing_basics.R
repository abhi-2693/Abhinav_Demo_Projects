###############################################
# ISB AMPBA - Term 1
# STATISTICAL ANALYSIS - 1 - Group Assignment
# Author : Group 11
# PG-ID	    | Name	            | Email
# --------------------------------------------------------------------
# 12420009	| Dacha Bhavana	    | Dacha_Bhavana_ampba2025W@isb.edu
# 12420054	| Kirthika Chandan	| Kirthika_Chandan_ampba2025W@isb.edu
# 12420055	| Abhinav Paul	    | Abhinav_Paul_ampba2025W@isb.edu
# 12420070	| Saketh Gutha	    | Saketh_Gutha_ampba2025W@isb.edu 
# --------------------------------------------------------------------
# Date : 06/05/2025
###############################################

# Load necessary libraries
# library(tidyverse)   # for dplyr, ggplot2, etc.
# library(skimr)       # for skim() - nice summary
# library(DataExplorer) # for quick automated EDA

# Set your working directory (optional if file path is complete)
# setwd("/Users/abhinavpaul/Desktop/Goals/Exec Edu/1. ISB AMPBA/Course Material/Term 1/3. Statistical Analysis 1/Assignment")

# Load the data
# data_org <- read_csv("SA1_Group_11.csv")

# Preview the data
# head(data_org)

# Check the structure of the dataset
# str(data_org)

# Summary statistics
# summary(data_org)

# Check missing values
# colSums(is.na(data_org))

# Alternatively, use skimr for a more detailed summary
# skim(data_org)

# Check for duplicate rows
# n_duplicates <- data_org %>% duplicated() %>% sum()
# cat("Number of duplicate rows:", n_duplicates, "\n")

# Check column types
# sapply(data_org, class)

# Basic data visualization for NA values
# plot_missing(data_org)

# Check for zero variance variables
# zero_var_cols <- data_org %>%
#   summarise(across(everything(), ~ n_distinct(.) == 1)) %>%
#   pivot_longer(everything(), names_to = "column", values_to = "zero_variance") %>%
#   filter(zero_variance == TRUE)

# if(nrow(zero_var_cols) > 0) {
#   print("Columns with zero variance:")
#   print(zero_var_cols$column)
# } else {
#   cat("No zero-variance columns found.\n")
# }

# Correlation plot for numeric variables
# data_org %>%
#   select(where(is.numeric)) %>%
#   cor(use = "pairwise.complete.obs") %>%
#   corrplot::corrplot(method = "color", type = "upper")

# Optional: Automated basic report
# create_report(data_org)

### clear all data from Memory
rm(list = ls())
cat("\014")
# if (!is.null(dev.list())) dev.off()

# Load the data
data_org <- read_csv("SA1_Group_11.csv")

#########################################################################
# Outlier Removal 
# GOP 3 year
summary(data_org$GOP_Year3) # this tells us the GOP data is in range of 10^5 Rs (Median) 

# checking for the last two max values
print(data_org[data_org$GOP_Year3 == max(data_org$GOP_Year3, na.rm=TRUE),]$GOP_Year3/10^5)
df_max_value <- head(data_org[data_org$GOP_Year3 < max(data_org$GOP_Year3, na.rm=TRUE), ],20)
df_less_max_row <- head(df_max_value[order(-df_max_value$GOP_Year3), ],1)
print(df_less_max_row$GOP_Year3/10^5)

# Looking at the last two rows and the context of the GOP_year3 column, we can see that the Max value is a incorrect data entry
# We know that the data is related to Small scale Units. 
# If a unit is generating 90810 Cr of Gross output in a year it is not a small-scale unit  

# removing the outlier GOP_Year3 row from the working sample
df_final <- data_org[data_org$GOP_Year3 < max(data_org$GOP_Year3, na.rm=TRUE), ]
# We cannot have initial Purchase of Machinery at 0 for any unit. These are bad data points
df_final <- df_final[df_final$ORI_PURC_VAL_PM > 0, ] 
# Replace the Market value = 0 with Re 1 as a substitute value
df_final$MKT_VAL_FA[df_final$MKT_VAL_FA < 1] <- 1
# De-duplication of the rows
df_final <- df_final[!duplicated(df_final), ]
nrow(df_final)

# #### 
# Q1
# #### 

sample_mean <- mean(df_final$GOP_Year3)
sample_stddev <- sd(df_final$GOP_Year3)
sample_size <- nrow(df_final)
CL <- 0.95
print(sample_size)
# sample size is large enough for us to consider the Sample mean distribution to be Normal.
# We can use Std Norm Dist as substitute for T dist for the MOE calculation even though the std.dev we are using is a sample estimate.
# this is because the Degree of Freedom in this Sample is 9998 which is large enough to make the approximation
# Impact : the MOE is less (but negligibly)
Z <- qnorm((1-CL)/2, lower.tail=FALSE) 
t <- qt(p=(1-CL)/2, df=sample_size-1, lower.tail=FALSE)
print(cat("Z, t are", Z, t))
# Calculate LB and UB
LB <- round((sample_mean - (t * sample_stddev / sqrt(sample_size)))/10^5, 2) ## Units in Lakhs
UB <- round((sample_mean + (t * sample_stddev / sqrt(sample_size)))/10^5, 2)## Units in Lakhs
# Create a table (data.frame)
ci_table <- data.frame(
  Metric = c("GOP_Year3 (Rs - in Lakhs)"),
  Lower_Bound = LB,
  Upper_Bound = UB
)
# Print the table
print(ci_table)

# #### 
# Q2
# #### 

metric1_def <- 'GOP_Year3 / MKT_VAL_FA (Asset productivity of the Year 3)'
df_final$Metric1 <- df_final$GOP_Year3 * 1.0 / df_final$MKT_VAL_FA
metric2_def <- 'MKT_VAL_FA / ORI_PURC_VAL_PM (ROI)'
df_final$Metric2 <- df_final$MKT_VAL_FA * 1.0 / df_final$ORI_PURC_VAL_PM
metric_def <- c(metric1_def, metric2_def)

# #### 
# Q3
# #### 

sample_mean_array <- c(mean(df_final$Metric1), mean(df_final$Metric2))
sample_stddev_array <- c(sd(df_final$Metric1), sd(df_final$Metric2))
sample_size <- nrow(df_final)
CL <- 0.99
Z <- qnorm((1-CL)/2, lower.tail=FALSE) 
t <- qt(p=(1-CL)/2, df=sample_size-1, lower.tail=FALSE)
print(cat("Z, t are", Z, t))
# Calculate LB and UB
LB <- round(sample_mean_array - 1.0 * (t * sample_stddev_array / sqrt(sample_size)), 2)
UB <- round(sample_mean_array + 1.0 * (t * sample_stddev_array / sqrt(sample_size)), 2)
# Create a table (data.frame)
ci_table <- data.frame(
  Metric = c("Metric1", "Metric2"),
  def = metric_def,
  Lower_Bound = LB,
  Upper_Bound = UB
)
# Print the table
print(ci_table)

# #### 
# Q4
# #### 
# a)
CL <- 0.995 # for both side combined 1% error
# Calculate total number of firms
total_firms <- nrow(df_final)
# Calculate number of SSSBE units
sssbe_firms <- nrow(df_final[df_final$UNIT_TYPE == 2, ])
# Calculate probability (sample proportion)
probability_sssbe <- sssbe_firms / total_firms
# Z-score for 95% confidence level
z_score <- qnorm(CL)  
# Standard error for proportion
se <- sqrt(probability_sssbe * (1 - probability_sssbe) / total_firms)
# Confidence Interval
lower_bound <- probability_sssbe - z_score * se
upper_bound <- probability_sssbe + z_score * se
cat("Probability that a randomly selected firm is a SSSBE unit:", round(probability_sssbe, 4), "\n")
cat("99% Confidence Interval (Z-based): (", round(lower_bound, 4), ",", round(upper_bound, 4), ")\n")
# b)
# Calculate the average of Metric1
metric1_avg <- mean(df_final$Metric1, na.rm = TRUE)
# Classify firms as Good or Bad
df_final$Performance <- ifelse(df_final$Metric1 > metric1_avg, "Good", "Bad")
# Calculate number of Good firms
good_firms <- nrow(df_final[df_final$Performance == "Good", ])
# Total number of firms
total_firms <- nrow(df_final)
# Sample proportion (probability) of Good firms
probability_good <- good_firms / total_firms
# Z-score for 95% confidence level
z_score <- qnorm(CL)
# Standard error for proportion
se <- sqrt(probability_good * (1 - probability_good) / total_firms)
# Confidence Interval
lower_bound <- probability_good - z_score * se
upper_bound <- probability_good + z_score * se
cat("Probability that a randomly selected firm is GOOD:", round(probability_good, 4), "\n")
cat("99% Confidence Interval (Z-based): (", round(lower_bound, 4), ",", round(upper_bound, 4), ")\n")
# c)
# Filter firms that are both SSSBE and Good
sssbe_good_firms <- nrow(df_final[df_final$UNIT_TYPE == 2 & df_final$Performance == "Good", ])
# Total number of firms
total_firms <- nrow(df_final)
# Probability calculation
probability_sssbe_good <- sssbe_good_firms / total_firms
# Z-score for 95% confidence level
z_score <- qnorm(CL)
# Standard error for proportion
se <- sqrt(probability_sssbe_good * (1 - probability_sssbe_good) / total_firms)
# Confidence Interval
lower_bound <- probability_sssbe_good - z_score * se
upper_bound <- probability_sssbe_good + z_score * se
cat("Probability that a randomly selected firm is BOTH SSSBE and GOOD:", round(probability_sssbe_good, 4), "\n")
cat("99% Confidence Interval (Z-based): (", round(lower_bound, 4), ",", round(upper_bound, 4), ")\n")
# d)
# Count total SSSBE units
sssbe_firms <- nrow(df_final[df_final$UNIT_TYPE == 2, ])
# Count SSSBE units with GOOD performance
sssbe_good_firms <- nrow(df_final[df_final$UNIT_TYPE == 2 & df_final$Performance == "Good", ])
# Conditional probability P(GOOD | SSSBE)
P_GOOD_given_SSSBE <- sssbe_good_firms / sssbe_firms
# P(BAD | SSSBE)
P_BAD_given_SSSBE <- 1 - P_GOOD_given_SSSBE
# Z-distribution 95% Confidence Interval for P(GOOD | SSSBE)
z_score <- qnorm(CL)  
se <- sqrt(P_GOOD_given_SSSBE * (1 - P_GOOD_given_SSSBE) / sssbe_firms)
lower_bound <- P_GOOD_given_SSSBE - z_score * se
upper_bound <- P_GOOD_given_SSSBE + z_score * se
cat("P(GOOD | SSSBE):", round(P_GOOD_given_SSSBE, 4), "\n")
cat("99% Confidence Interval for P(GOOD | SSSBE): (", round(lower_bound, 4), ",", round(upper_bound, 4), ")\n")
lower_bound <- P_BAD_given_SSSBE - z_score * se
upper_bound <- P_BAD_given_SSSBE + z_score * se
cat("P(BAD | SSSBE):", round(P_BAD_given_SSSBE, 4), "\n")
cat("99% Confidence Interval for P(GOOD | SSSBE): (", round(lower_bound, 4), ",", round(upper_bound, 4), ")\n")
########
# Conclusion
# The probability that an SSSBE unit is GOOD is relatively low at 14.38% with CI [12.42, 16.34] at 99% (both sided) Confidence Level
# The probability that an SSSBE unit is BAD is much higher at 85.62% with CI [83.66, 87.58] at 99% (both sided) Confidence Level
# This indicates that, based on the probabilities calculated above, the performance of SSSBE units in terms of being GOOD is quite low
########

# #### 
# Q5
# #### 
table(factor(df_final$VOE_Year3)) # value unique table
mu = 87300 #Given the Population Mean 
sample_size <- length(df_final$VOE_Year3)
sample_stddev <- sd(df_final$VOE_Year3)
sample_mean <- mean(df_final$VOE_Year3)
df <- sample_size-1
t <- (sample_mean - mu)/(sample_stddev/sqrt(sample_size))
print(cat("pop_mean_sigma :", mu, ""))
print(cat("sample_size :", sample_size, ""))
print(cat("sample_mean :", sample_mean, ""))
print(cat("sample_stddev :", sample_stddev, ""))
print(cat("Degree of Freedom :", df, ""))
print(cat("t-value :", t, ""))

# ----------Confusion Matrix-----------
#           | <=87300          | > 87300
# Flag      | TRUE            | Beta
# Not Flag  | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_mean <= 87300
# H1 : sample_mean > 87300 (We want to test if the Mean is greater than the given Population mean)
p <- 1 - pt(t, df=df, lower.tail = TRUE) # Right tail Test as Null Hypothesis is to have Sample mean <= 87300 
print(cat("P-Value :", p, ""))
CL<- 0.99 # Both sided error
if (p<(1-CL)/2) {print("H0 is rejected")} else {print("Failed to rejected H0")}

########
# Conclusion 
# Based on the sample we can not conclude that the mean is equal to (or less than) 87300.
########
# we also have this function to do One sided T test - Used to verify our Answer
# t.test(df_final$VOE_Year3, alternative = 'greater', mu = mu, conf.level=CL)

# #### 
# Q6
# ####
sample_size_array <- c(nrow(df_final[df_final$UNIT_TYPE==1,]), nrow(df_final[df_final$UNIT_TYPE==2,]))
sample_proportion_array <- sample_size_array/length(df_final$Metric2)
print(cat("sample_size_array (1=SSI and 2=SSSBE) :", sample_size_array, ""))
print(cat("sample_proportion_array (1=SSI and 2=SSSBE) :", sample_proportion_array, ""))
# ----------Confusion Matrix-----------
#           | >=0.25          | < 0.25
# Not Flag  | TRUE            | Beta
# Flag      | Alpha (P-value) | TRUE
# -------------------------------------
# To decide on whether to go for Special Incentive is needed we check for below Null Hypothesis
# Null Hypothesis is We assume that Units type does need Special Incentive (i.e <0.25)
# H0 : sample_prop >= 0.25 (for both SSSBE and SSI individually) (Rejecting this means the Units type has population proportion less than 0.25 individually and needs special Incentive)
# H1 : sample_prop < 0.25 (for both SSSBE and SSI individually)
population_proportion <- 0.25 # Hypothesis to test
sample_sd_array <- sqrt(population_proportion*(1-population_proportion)/sum(sample_size_array))
print(cat("sample_sd :", sample_sd_array, ""))
CL = 0.99 # Both sided error
Z_score <- (sample_proportion_array-population_proportion)/sample_sd_array
p <- pnorm(Z_score, lower.tail = TRUE) # Left Tail Test 
print(cat("Z_score (1=SSI and 2=SSSBE) :", Z_score, " against Z-critical", qnorm((1-CL)/2, lower.tail = TRUE), ""))
print(cat("P-value (1=SSI and 2=SSSBE) :", p, "against P-Critical", ((1-CL)/2), ""))
decision <- data.frame(unit_type=c('SSI', 'SSSBE'), 
                       sample_proportion_array = sample_proportion_array,
                       p_value = p,
                       H0_sample_prop = ifelse(p < (1 - CL)/2, "H0 is rejected", "Failed to reject H0")
                       )
print(decision)

########
# Conclusion 
# for SSI   - We can not conclude with statistical significance that the proportion is less than 0.25 and need Special Incentive
# for SSSBE - We can conclude with statistical significance that the proportion is less than 0.25 and needs Special Incentive
########
# We have prop.test to do the same analysis - Used to check our answer
# prop.test(x = sum(df_final$UNIT_TYPE==1), n = nrow(df_final), conf.level=CL, correct = FALSE, p = 0.25, alternative = "less")
# prop.test(x = sum(df_final$UNIT_TYPE==2), n = nrow(df_final), conf.level=CL, correct = FALSE, p = 0.25, alternative = "less")

# #### 
# Q7
# #### 
df_sssbe <- df_final[df_final$UNIT_TYPE==2,]
sample_gender_dist_array <- c(nrow(df_sssbe[df_sssbe$MAN_BY==1,]), nrow(df_sssbe[df_sssbe$MAN_BY==2,]))
sample_gender_proportion_array <- sample_gender_dist_array/nrow(df_sssbe)
combined_sample_good_proportion <- sum(sample_gender_dist_array)/(2*nrow(df_sssbe))
print(cat("sample_gender_dist_array (1=MALE and 2=FEMALE) :", sample_gender_dist_array, ""))
print(cat("sample_gender_proportion_array (1=MALE and 2=FEMALE) :", sample_gender_proportion_array, ""))
print(cat("combined_sample_good_proportion :", combined_sample_good_proportion, ""))
# Let the Null Hypothesis be that male chauvinists are correct i.e. P(Female|SSSBE) < P(Male|SSSBE)
# ----------Confusion Matrix-----------
#               | <=0             | > 0
# Male - Female | TRUE            | Beta
# Male - Female | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_proportion_male - sample_proportion_female <= 0
# H1 : sample_proportion_male - sample_proportion_female  > 0
Z <- ((sample_gender_proportion_array[1] - sample_gender_proportion_array[2]) - 0)/sqrt(combined_sample_good_proportion*(1-combined_sample_good_proportion)*(2/nrow(df_sssbe)))
CL = 0.99 # Both sided error
z_critical <- 1-qnorm((1-CL)/2, lower.tail = TRUE) # Right Tailed Test
print(cat("Z-value :", Z, "against Z-Critical :", z_critical, ""))
if (Z>z_critical) {print("H0 is rejected")} else {print("Failed to rejected H0")}
p <- 1-pnorm(Z, mean=0, lower.tail=TRUE) # Right Tail Test
print(cat("P-Value :", p, "against P-Critical :", (1-CL)/2, ""))
if (p<(1-CL)/2) {print("H0 is rejected")} else {print("Failed to rejected H0")}

########
# Conclusion 
# Since p-value ≈ 0 (Z> Z-critical), you strongly reject H0.
# This is very strong statistical evidence that a larger proportion of SSSBE units are managed by men compared to women.
########

# We also have prop.test to do above analysis - Used to check our answers
# prop.test(
#   x = c(nrow(df_sssbe[df_sssbe$MAN_BY==1,]), nrow(df_sssbe[df_sssbe$MAN_BY==2,])),
#   n = c(nrow(df_sssbe), nrow(df_sssbe)),
#   conf.level=0.99,
#   correct = FALSE,
#   alternative = 'greater'
# )

# #### 
# Q8
# #### 
print(cat("definition of the metric :", metric2_def, ""))
summary(df_final$Metric2) 
# when UNIT_TYPE==2 its SSSBE and UNIT_TYPE==1 its SSI
sample_mean_array <- c(mean(df_final$Metric2[df_final$UNIT_TYPE==1]), mean(df_final$Metric2[df_final$UNIT_TYPE==2]))
sample_sd_array <- c(sd(df_final$Metric2[df_final$UNIT_TYPE==1]), sd(df_final$Metric2[df_final$UNIT_TYPE==2]))
sample_size_array <- c(length(df_final$Metric2[df_final$UNIT_TYPE==1]), length(df_final$Metric2[df_final$UNIT_TYPE==2]))
sample_var_array <- sample_sd_array^2/sample_size_array
print(cat("sample_size (1=SSI and 2=SSSBE) :", sample_size_array, ""))
print(cat("sample_mean (1=SSI and 2=SSSBE) :", sample_mean_array, ""))
print(cat("sample_stddev (1=SSI and 2=SSSBE) :", sample_sd_array, ""))
print(cat("sample_var (1=SSI and 2=SSSBE) :", sample_var_array, ""))
# When comparing the the performance between SSSBE and SSI - units types
# if they are same then the difference between two population means will be 0 
# Let the Null Hypothesis be that SSSBE is Better than SSI
# ----------Confusion Matrix-----------
#             | <=0             | > 0
# SSI - SSSBE | TRUE            | Beta
# SSI - SSSBE | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_mean_array[1](SSI)-sample_mean_array[2](SSSBE) <=0 (Rejecting this means SSSBE is better ROI than SSI)
# H1 : sample_mean_array[1](SSI)-sample_mean_array[2](SSSBE) > 0 
df <- (sum(sample_var_array)^2)/sum((sample_var_array)^2/(sample_size_array-1))
t <- ((sample_mean_array[1]-sample_mean_array[2]) - 0)/sqrt(sum(sample_var_array))
print(cat("degree of freedom :", df, ""))
print(cat("t-value :", t, ""))
CL = 0.99 # Both sided error
p <- 1-pt(t, df, lower.tail=TRUE) # Right Tailed Test (to check for SSI - SSSBE <=0)
print(cat("P-Value :", p, "against the p-critical ", (1-CL)/2, ""))
if (p<(1-CL)/2) {print("H0 is rejected")} else {print("Failed to rejected H0")}

# Now, Let the Null Hypothesis be that SSI is Better than SSSBE (reversed case)
# ----------Confusion Matrix-----------
#             | <=0             | > 0
# SSSBE - SSI | TRUE            | Beta
# SSSBE - SSI | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_mean_array[2](SSSBE)-sample_mean_array[1](SSI) <=0 (Rejecting this means SSSBE is better ROI than SSI)
# H1 : sample_mean_array[2](SSSBE)-sample_mean_array[1](SSI) > 0 
df <- (sum(sample_var_array)^2)/sum((sample_var_array)^2/sample_size_array)
t <- ((sample_mean_array[2]-sample_mean_array[1]) - 0)/sqrt(sum(sample_var_array))
print(cat("degree of freedom :", df, ""))
print(cat("t-value :", t, ""))
CL = 0.99 # Both sided error
p <- 1-pt(t, df, lower.tail=TRUE) # Right Tailed Test (to check for SSSBE - SSI <=0)
print(cat("P-Value :", p, "against the p-critical ", (1-CL)/2, ""))
if (p<(1-CL)/2) {print("H0 is rejected")} else {print("Failed to rejected H0")}

## NOTE : We could have done the above with left tailed T test as well for the 1st Null Hypothesis of SSI-SSSBE metric

########
# Conclusion 
# Since I am not able to reject the H0 (mean of ROI metric for SSI to be Less than SSSBE or the reverse case)
# we can not conclude on the population mean difference of SSSBE and SSI with our given Sample data
########

# we also have this function to do One sided T test - Used to verify our Answer
# t.test(Metric2 ~ UNIT_TYPE, data = df_final, alternative="greater")
# t.test(Metric2 ~ UNIT_TYPE, data = df_final, alternative="less")

# #### 
# Q9
# #### 
# Part 1 - Metric 2
summary(df_final$Metric1) 
print(cat("definition of the metric :", metric1_def))
# when UNIT_TYPE==2 its SSSBE and UNIT_TYPE==1 its SSI
sample_mean_array <- c(mean(df_final$Metric1[df_final$UNIT_TYPE==1]), mean(df_final$Metric1[df_final$UNIT_TYPE==2]))
sample_size_array <- c(nrow(df_final[df_final$UNIT_TYPE==1,]), nrow(df_final[df_final$UNIT_TYPE==2,]))
df_final$good_SSI <- 0 # defining good performance units (Type : SSI)
df_final$good_SSI[df_final$UNIT_TYPE==1 & df_final$Metric1 > sample_mean_array[1]] <- 1
df_final$good_SSSBE <- 0 # defining good performance units (Type : SSSBE)
df_final$good_SSSBE[df_final$UNIT_TYPE==2 & df_final$Metric1 > sample_mean_array[2]] <- 1
sample_good_proportion_array <- c(sum(df_final$good_SSI), sum(df_final$good_SSSBE))/sample_size_array
sample_bad_proportion_array <- 1 - sample_good_proportion_array
combined_sample_good_proportion <- (sum(df_final$good_SSI)+sum(df_final$good_SSSBE))/sum(sample_size_array)
print(cat("sample_size_array (1=SSI and 2=SSSBE) :", sample_size_array, ""))
print(cat("sample_good_proportion_array (1=SSI and 2=SSSBE) :", sample_good_proportion_array, ""))
print(cat("sample_bad_proportion_array (1=SSI and 2=SSSBE) :", sample_bad_proportion_array, ""))
print(cat("combined_sample_good_proportion :", combined_sample_good_proportion, ""))
# When comparing the proportion of good performing units between SSSBE and SSI - units types
# if the proportion are assumed to be same then the difference between two population proportion will be 0
# Lets Take Null Hypothesis - the good proportion in SSI is better than SSSBE
# ----------Confusion Matrix-----------
#             | <=0             | > 0
# SSSBE - SSI | TRUE            | Beta
# SSSBE - SSI | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_good_proportion_array[2](SSSBE)-sample_good_proportion_array[1](SSI) <=0 (Rejecting this means proportion of good performing SSI units is more than SSSBE)
# H1 : sample_good_proportion_array[2](SSSBE)-sample_good_proportion_array[1](SSI) > 0 
# proportion can not be tested based on the t-Distribution cause the parent distribution is not Normal but Binomial
Z <- ((sample_good_proportion_array[1]-sample_good_proportion_array[2]) - 0)/( sqrt(combined_sample_good_proportion * (1 - combined_sample_good_proportion) * sum(1/sample_size_array)))
CL = 0.99 # both sided error
z_critical <- qnorm((1-CL)/2, lower.tail = FALSE) # Right tail Test 
print(cat("Z-value :", Z, "against Z-Critical :", z_critical, ""))
if (Z>z_critical) {print("H0 is rejected")} else {print("Failed to rejected H0")}
p <- 1-pnorm(Z, mean=0, lower.tail=TRUE) # Right Tail Test as our Null Hypothesis was that the difference is <=0 
print(cat("P-Value :", p, "and Alpha needed is ",(1-CL)/2))
if (p<((1-CL)/2)) {print("H0 is rejected")} else {print("Failed to rejected H0")}

#We also have prop.test to do above analysis - Used to check our answers
# prop.test(
#   x = c(sum(df_final$good_SSI, na.rm = TRUE), sum(df_final$good_SSSBE, na.rm = TRUE)),
#   n = sample_size_array,
#   conf.level=CL,
#   correct = FALSE,
#   alternative = 'greater'
# )

# Part 1 - Metric 2
summary(df_final$Metric2) 
print(cat("definition of the metric :", metric2_def))
# when UNIT_TYPE==2 its SSSBE and UNIT_TYPE==1 its SSI
sample_mean_array <- c(mean(df_final$Metric2[df_final$UNIT_TYPE==1]), mean(df_final$Metric2[df_final$UNIT_TYPE==2]))
sample_size_array <- c(nrow(df_final[df_final$UNIT_TYPE==1,]), nrow(df_final[df_final$UNIT_TYPE==2,]))
df_final$good_SSI <- 0 # defining good performance units (Type : SSI)
df_final$good_SSI[df_final$UNIT_TYPE==1 & df_final$Metric2 > sample_mean_array[1]] <- 1
df_final$good_SSSBE <- 0 # defining good performance units (Type : SSSBE)
df_final$good_SSSBE[df_final$UNIT_TYPE==2 & df_final$Metric2 > sample_mean_array[2]] <- 1
sample_good_proportion_array <- c(sum(df_final$good_SSI), sum(df_final$good_SSSBE))/sample_size_array
sample_bad_proportion_array <- 1 - sample_good_proportion_array
combined_sample_good_proportion <- (sum(df_final$good_SSI)+sum(df_final$good_SSSBE))/sum(sample_size_array)
print(cat("sample_size_array (1=SSI and 2=SSSBE) :", sample_size_array, ""))
print(cat("sample_good_proportion_array (1=SSI and 2=SSSBE) :", sample_good_proportion_array, ""))
print(cat("sample_bad_proportion_array (1=SSI and 2=SSSBE) :", sample_bad_proportion_array, ""))
print(cat("combined_sample_good_proportion :", combined_sample_good_proportion, ""))
# When comparing the proportion of good performing units between SSSBE and SSI - units types
# if the proportion are assumed to be same then the difference between two population proportion will be 0
# Lets Take Null Hypothesis - the good proportion in SSI is better than SSSBE
# ----------Confusion Matrix-----------
#             | <=0             | > 0
# SSSBE - SSI | TRUE            | Beta
# SSSBE - SSI | Alpha (P-value) | TRUE
# -------------------------------------
# H0 : sample_good_proportion_array[2](SSSBE)-sample_good_proportion_array[1](SSI) <=0 (Rejecting this means proportion of good performing SSI units is more than SSSBE)
# H1 : sample_good_proportion_array[2](SSSBE)-sample_good_proportion_array[1](SSI) > 0 (Rejecting this means we cannot reject the H0 of proportion of good performing SSI units is more than SSSBE)
# proportion can not be tested based on the t-Distribution cause the parent distribution is not Normal but Binomial
Z <- ((sample_good_proportion_array[1]-sample_good_proportion_array[2]) - 0)/( sqrt(combined_sample_good_proportion * (1 - combined_sample_good_proportion) * sum(1/sample_size_array)))
CL = 0.99 # both sided error
z_critical <- qnorm((1-CL)/2, lower.tail = FALSE) # Right tail Test 
print(cat("Z-value :", Z, "against Z-Critical :", z_critical, ""))
if (Z>z_critical) {print("H0 is rejected")} else {print("Failed to rejected H0")}
p <- 1-pnorm(Z, mean=0, lower.tail=TRUE) # Right Tail Test as our Null Hypothesis was that the difference is <=0 
print(cat("P-Value :", p, "and Alpha needed is ",(1-CL)/2))
if (p<((1-CL)/2)) {print("H0 is rejected")} else {print("Failed to rejected H0")}

#We also have prop.test to do above analysis - Used to check our answers
# prop.test(
#   x = c(sum(df_final$good_SSI, na.rm = TRUE), sum(df_final$good_SSSBE, na.rm = TRUE)),
#   n = sample_size_array,
#   conf.level=CL,
#   correct = FALSE,
#   alternative = 'greater'
# )
########
# Conclusion
# for Both metric defined we are able to Reject the Hypothesis that performance of SSI is better than SSSBE 
# for Both the metric We have statistically significance that SSSBE performance is better than SSI
# NOTE : In Q6 we also had established the Population Proportion of SSSBE is < 0.25
#        , and even with such low population representation the Unit type is performing really well
########

# #### 
# Q10
# #### 
summary(df_final$GOP_Year2)
summary(df_final$GOP_Year1)
df_gop_2_1 <- na.omit(df_final[, c("GOP_Year2", "GOP_Year1")])
df_gop_2_1$GOP_Y1_Y2_delta <- df_gop_2_1$GOP_Year2 - df_gop_2_1$GOP_Year1
sample_delta_mean <- mean(df_gop_2_1$GOP_Y1_Y2_delta)
sample_delta_sd <- sd(df_gop_2_1$GOP_Y1_Y2_delta)
sample_size <- length(df_gop_2_1$GOP_Y1_Y2_delta)
delta_mu <- 0 # for the condition where GOP_Y2 = GOP_Y1
print(cat("sample_size :", sample_size, ""))
print(cat("sample_delta_mean :", sample_delta_mean, ""))
print(cat("sample_delta_stddev :", sample_delta_sd, ""))

# Lets assume the contention on GOP_Y2 being larger than GOP_Y1 be True
# So the Null Hypothesis is GOP_Y2 > GOP_Y1
# ----------Confusion Matrix-----------------------------------
#                 | <=0               | > 0
# GOP_Y2 - GOP_Y1 | TRUE              | Beta
# GOP_Y2 - GOP_Y1 | Alpha (P-value)   | TRUE
# -------------------------------------------------------------
# H0 : mean delta (GOP Y2-Y1) <= 0 (Rejecting this means avg GOP Y2 is mostly better then avg GOP Y1)
# H1 : mean delta (GOP Y2-Y1) > 0 
df <- sample_size - 1
t <- (sample_delta_mean - delta_mu)/(sample_delta_sd/sqrt(sample_size))
print(cat("t-value :", t, ""))
print(cat("degree of freedom :", df, ""))
p <- 1-pt(t, df=df, lower.tail = TRUE) # Right tail Test
CL<- 0.99 # both sided error
print(cat("P-Value :", p, " aginast the P-Critical", (1-CL)/2, ""))
if (p<((1-CL)/2)) {print("H0 is rejected")} else {print("Failed to rejected H0")}

########
# Conclusion
# We can not conclude that the GOP_Y2 is always more than GOP_Y1 with any statistical significance at 99% CL
########

# we also have this function to do One sided T test - Used to verify our Answer
# t.test(df_gop_2_1$GOP_Y1_Y2_delta, alternative = 'greater', mu = delta_mu, conf.level=CL)
# we can also do Paired t-test
# t.test(df_gop_2_1$GOP_Year2, df_gop_2_1$GOP_Year1, paired = TRUE, alternative = "greater")

