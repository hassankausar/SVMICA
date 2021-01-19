#--------------------------- PRE-PROCESSING THE DATA ---------------------------

# Installing the packages required for pre-processing and to work with the model later.
# PLEASE NOTE NOT ALL PACKAGES ARE REQUIRED FOR THIS MODEL SPECIFICALLY
# AS I AM WORKING WITH KNN AND SVM TOO I LISTED BOTH HERE.
install.packages("dplyr")
install.packages("magrittr")
install.packages("knitr")
install.packages("reshape")
install.packages("caret")
install.packages("e1071")
install.packages("kernlab")
install.packages("ggplot2")
install.packages("gbm")
install.packages("pROC")
install.packages("tidyverse")
install.packages("dlstats")
install.packages("ROC")
install.packages("pkgsearch")
install.packages("ROCR")


# Initialising the libraries.
library(dplyr)
library(magrittr)
library(knitr)
library(reshape)
library(caret)
library(e1071)
library(kernlab)
library(ggplot2)
library(datasets)
library(gbm)
library(pROC)
library(tidyverse)
library(dlstats)
library(pkgsearch)
rocPkg <-  pkg_search(query = "ROC", size = 200)
library(ROCR)
#-------- Initialising the directory and replacing the variable names  ---------


# Initialising the directory and using the data for pre-process
# WORKING WITH THIS DATA FROM HOME
setwd("C:/Users/Hassan/Desktop/ICA WORK SUBMISSION/Hassan Akhtar Kausar - V8039087/Part 1 - Video, Application/SVMICA")

#setwd("C/:Users/V8039087/Desktop/MACHINE LEARNING FULL ICA WORK/KNNICA-master/KNNICA-master")

# WORKING WITH THIS DATA FROM UNI
#setwd("U:/YEAR 3 COURSE WORK/ML ICA")




# Reading the file which contains the data and loading it to the workspace.
BreastCancer <-
  read.csv(file = "BreastCancerOriginal.data",
           stringsAsFactors = FALSE,
           header = TRUE)



# Changing variable names. Currently set from X1 to X5.
BreastCancer <-  rename(
  BreastCancer,
  c(
    X1000025 = "Patient_ID",
    X5 = "Clump_Thickness",
    X1 = "Uniformity_of_Cell_Size",
    X1.1 = "Uniformity_of_Cell_Shape",
    X1.2 = "Marginal_Adhesion",
    X2 = "Single_Epithelial_Cell_Size",
    X1.3 = "Bare_Nuclei",
    X3 = "Bland_Chromatin",
    X1.4 = "Normal_Nucleoli",
    X1.5 = "Mitoses",
    X2.1 = "Class"
  )
)


# Checking the data summary which includes the Mean, Mode and Median.
summary(BreastCancer)


# Identifying the class type of Bare Nuclei Variable so then I can proceed with the data replacing.
class(BreastCancer$Bare_Nuclei)

# Summary() doesn't give any information at all for the Bare Nuclei variable apart from the length and the class (which are 'character').
# Replacing to factor will help to get the mode as we will be working with that later.
BreastCancer$Bare_Nuclei <- as.factor(BreastCancer$Bare_Nuclei)


# Checking the summary for BareNuclei variable where the class is 2 and 4.
# This should give us the mode for both which later will be used to replace the data.
summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "4"])
summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "2"])



#----------------------- Deleting and Replacing data  -------------------------


# Deleting the rows that are not required due to I identified similar data to those.
BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617), ] %>% head()
BreastCancer <-
  BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617), ]



# Replacing the missing data by its mode for the Bare Nuclei variable.

# If the class is 4 the data will be replaced with 10.
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "4"] <- "10"

# If the class is 2 the data will be replaced with 1.
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "2"] <- "1"


#Checking the summary again to see everything is same or if it has changed.
summary(BreastCancer)

# Making sure the data is still classified as factor before checking it.
BreastCancer$Bare_Nuclei <- as.factor(BreastCancer$Bare_Nuclei)

# An alternative to Summary, it displays the class type of each variable along with top 10 instances of data.
str(BreastCancer)






#-------------- Data Slicing, Testing,  Training and Correlation --------------

# Data slicing is a step to split data into train and test set.
# The set.seed function in this model is used since cross validation randomly assigns rows to each fold
# but we want to be able to reproduce our model
set.seed(206)

# The method createDataPartition() is used for partitioning our data into train and test set.
# The “Y” parameter takes the value of variable according to which data needs to be partitioned.
# In my case, target variable is the Class Variable.
# P shows the percentage of the split which should stand for 70%.
BreastCancerInTrain <-
  createDataPartition(y = BreastCancer$Class, p = 0.7, list = FALSE)

# The training data is the 70% of the data assigned above in the BreastCancerInTrain that I am training.
Training <- BreastCancer[BreastCancerInTrain, ]

# The testing data is the 30% of the data remaining from BreastCancerInTrain which is going to be tested.
Testing <- BreastCancer[-BreastCancerInTrain, ]

#Checking how many observations there are being trained. It should be either 482 or 483 because that would be the 70% of 689~.
dim(Training)

#Checking how many observations there are being tested. It should be either 206 or 207 because that would be the 30% of 689~.
dim(Testing)

#Checking if there is any empty data in our dataset.
anyNA(BreastCancer)

# Checking the summary again to make sure everything is correct before proceeding.
summary(BreastCancer)


# The code below will convert the training data frame’s “Class” variable to a factor variable.
# It was previously done too using different technique for the BreastCancer file.
Training[["Class"]] = factor(Training[["Class"]])

# Before we train our model, we will implement the trainControl() method.
# This will control all the computational overheads so that we can use the train() function provided
# by the caret package. The training method will train our data on different algorithms.
# The resampling method:"repeatedcv" is used for repeated training/test splits.
# Number: The number of folds or number of resampling iterations
# Repeats: The number of complete sets of folds to compute
# In 5 repeats of 10 fold CrossValidation, we’ll perform the average of 5 error terms obtained by performing 10 fold CV five times.
TrainControl <-
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 5)

# Test for association between paired samples, using one of Pearson's product moment correlation coefficient
BreastCancer.ctest2 <-
  cor.test(
    BreastCancer$Uniformity_of_Cell_Size,
    BreastCancer$Class,
    conf.level = 0.95,
    alternative = "two.sided"
  )
BreastCancer.ctest2








# ----------------- SVM Classifier using Linear Kernel --------------------
#-------------------- SVM LINEAR CLASSIFIER --------------------


# The following example variables are normalised to make their scale comparable.
# This is automatically done before building the SVM classifier by setting the option preProcess = c("center","scale").
# The train function supports fitting 100s of different models which can be easily specified using the Method argument which in my case is SVMLINEAR
# This is provided by the caret package . The trControl parameter controls the parameter for cross validation.
svm_Linear <- train(
  Class ~ .,
  data = Training,
  method = "svmLinear",
  trControl = TrainControl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)

# Displaying the SVM_LINEAR function used above.
svm_Linear



# HERE I WILL BE USING TWO DIFFERENT TECHNIQUES TO FIND THE PREDICTIONS AND FIND IF THEY RETURN THE SAME OR NOT.

# Prediction 1
# evaluating the performance for this model
SVMTestPred <- predict(svm_Linear, newdata = Testing)
# Displaying the predicition
SVMTestPred

# ConfusionMatrix 1
# Using confusion Matrix to get the results.
ConfusionMatrix = confusionMatrix(table(SVMTestPred, Testing$Class))
ConfusionMatrix

# Prediction 1
table(SVMTestPred, Testing$Class)

# Prediction 1
agreement <- SVMTestPred == Testing$Class

# Prediction 1
table(agreement)

# Prediction 1
prop.table(table(agreement))

plot(agreement)


#---------------- Different way of working with Test Prediction ----------------


# UNCOMMENT THE FOLLOWING LINES TO TEST.




#TestPrediction2 = predict(svm_Linear, Testing)
#TestPrediction2
#ConfusionMatrix2 = confusionMatrix(as.factor(Testing$Class), as.factor(TestPrediction2))
#ConfusionMatrix2
#table(TestPrediction2, Testing$Class)
#agreement <- TestPrediction2 == Testing$Class
#table(agreement)
#prop.table(table(agreement))





#-------------------- SVM LINEAR GRID --------------------

grid <-
  expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5))

# The tuning parameter C which is also known as Cost checks for possible misclassifications.
# The higher the value of C, the less likely it is that the SVM algorithm will misclassify a point.
# By default caret builds the SVM linear classifier using C = 1.
# It’s possible to automatically compute SVM for different values of `C and to choose the optimal one that maximize the model cross-validation accuracy.
# The following R code compute SVM for a grid values of C and choose automatically the final model for predictions:

svm_Linear_Grid <- train(
  Class ~ .,
  data = Training,
  method = "svmLinear",
  trControl = TrainControl,
  preProcess = c("center", "scale"),
  tuneGrid = grid,
  tuneLength = 10
)

# Displaying the SVM_LINEAR_GRID function used above.
svm_Linear_Grid

# Plotting the SVM Linear Grid so it is easily readable.
plot(svm_Linear_Grid)


# Predicting and evaluating the performance for this model
TestPrediction_grid <- predict(svm_Linear_Grid, newdata = Testing)

# Displaying the predicition
TestPrediction_grid

# Using confusion Matrix to get the results.
confusionMatrix(table(TestPrediction_grid, Testing$Class))













# ----------------- SVM Classifier using Non-Linear Kernel --------------------

#-------------------- SVM RADIAL CLASSIFIER --------------------

# Data slicing is a step to split data into train and test set.
# The set.seed function in this model is used since cross validation randomly assigns rows to each fold
# but we want to be able to reproduce our model
set.seed(206)

# The tuning parameter C which is also known as Cost checks for possible misclassifications.
# The higher the value of C, the less likely it is that the SVM algorithm will misclassify a point.
# As I am working with the Radial Classifier the method I am using is the SVMRADIAL
# All other details are same as Linear Model.
svm_Radial <- train(
  Class ~ .,
  data = Training,
  method = "svmRadial",
  trControl = TrainControl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)

# Displaying the SVM_RADIAL function used above.
svm_Radial

# Plotting the SVM Radial so it is easily readable.
plot(svm_Radial)

# Predicting the radial method
TestPrediction_Radial <- predict(svm_Radial, newdata = Testing)

# Displaying the predicition
TestPrediction_Radial

# Using confusion Matrix to get the results.
confusionMatrix(table(TestPrediction_Radial, Testing$Class))


#-------------------- SVM RADIAL GRID --------------------

grid_Radial <-
  expand.grid(
    sigma = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1),
    C =     c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1)
  )

svm_Radial_Grid <- train(
  Class ~ .,
  data = Training,
  method = "svmRadial",
  trControl = TrainControl,
  preProcess = c("center", "scale"),
  tuneGrid = grid_Radial,
  tuneLength = 10
)

# Displaying the SVM_RADIAL_Grid function used above.
svm_Radial_Grid

plot(svm_Radial_Grid)


# evaluating the performance for radial grid search
TestPrediction_Radial_Grid <-
  predict(svm_Radial_Grid, newdata = Testing)

# Displaying the predicition
TestPrediction_Radial_Grid

# Using confusion Matrix to get the results.
confusionMatrix(table(TestPrediction_Radial_Grid, Testing$Class))











#------------------ LINEAR KERNEAL FUNCTION-------------------

# Data slicing is a step to split data into train and test set.
# The set.seed function in this model is used since cross validation randomly assigns rows to each fold
# but we want to be able to reproduce our model
set.seed(206)

# The tuning parameter C which is also known as Cost checks for possible misclassifications.
# The higher the value of C, the less likely it is that the SVM algorithm will misclassify a point.
# By default caret builds the SVM linear classifier using C = 1.
# It’s possible to automatically compute SVM for different values of `C and to choose the optimal one that maximize the model cross-validation accuracy.
# The following code compute KSVM for values of C and choose automatically the final model for predictions:
svm_Kernel <- ksvm (
  Class ~ .,
  data = Training,
  method = "svmRadial",
  trControl = TrainControl,
  preProcess = c("center", "scale"),
  tuneGrid = grid_Radial,
  tuneLength = 10,
  kernel = "vanilladot"
)

# Displaying the SVM_KERNEL function used above.
svm_Kernel

# evaluating the performance for this model
Kernel_Prediction <- predict(svm_Kernel, newdata =  Testing)

# Displaying the predicition
Kernel_Prediction

# Using confusion Matrix to get the results.
confusionMatrix(table(Kernel_Prediction, Testing$Class))

# Agreement is used to verify that the accuracy given by the confusion matrix is correct and that we are reading the correct analysis.
agreement <- Kernel_Prediction == Testing$Class

# The table function display the correct observations and the incorrect ones and shows how many of each there were
table(agreement)

# This shows the percentage of accurate and inaccurate data.
prop.table(table(agreement))



#---------------------Gaussian Radial Basis Function Kernel---------------------
# Improving performance for the linear kernel using Gaussian Radial Basis Function Kernel
Kernel_classifier_rbf <-  ksvm(Class ~ ., data = Training, kernel = "rbfdot")
Kernel_predictions_rbf <- predict(Kernel_classifier_rbf, newdata = Testing)

# Displaying the KERNEL Predction function used above.
Kernel_predictions_rbf

# Using confusion Matrix to get the results.
confusionMatrix(table(Kernel_predictions_rbf, Testing$Class))

# Agreement is used to verify that the accuracy given by the confusion matrix is correct and that we are reading the correct analysis.
agreement_rbf <- Kernel_predictions_rbf == Testing$Class

# The table function display the correct observations and the incorrect ones and shows how many of each there were.
table(agreement_rbf)

# This shows the percentage of accurate and inaccurate data.
prop.table(table(agreement_rbf))

#use roc to plot the accuracies
