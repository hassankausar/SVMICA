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


#-------- Initialising the directory and replacing the variable names  ---------


# Initialising the directory and using the data for pre-process
# WORKING WITH THIS DATA FROM HOME
setwd("C:/Users/Hassan/Desktop/Machine Learning ICA work/SVMICA")



# WORKING WITH THIS DATA FROM UNI
#setwd("U:/YEAR 3 COURSE WORK/ML ICA")



# Reading the file which contains the data and loading it to the workspace.
BreastCancer <-  read.csv(file = "BreastCancerOriginal.data", stringsAsFactors = FALSE, header = TRUE)



# Changing variable names. Currently set from X1 to X5.
BreastCancer <-  rename(BreastCancer, 
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
    ))


# Checking the data summary which includes the Mean, Mode and Median.
summary(BreastCancer)


# Identifying the class type of Bare Nuclei Variable so then I can proceed with the data replacing.
class(BreastCancer$Bare_Nuclei)

# Summary() doesn't give any information at all for the Bare Nuclei variable apart from the length and the class (which are 'character').
# Replacing to factor will help to get the mode as we will be working with that later.
BreastCancer$Bare_Nuclei <- as.factor(BreastCancer$Bare_Nuclei)


summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "4"])
summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "2"])



#----------------------- Deleting and Replacing data  -------------------------


# Deleting the rows that are not required due to I identified similar data to those.
BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617),] %>% head()
BreastCancer <-BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617),]



# Replacing the missing data by its mode for the Bare Nuclei variable.

# If the class is 4 the data will be replaced with 10.
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" & BreastCancer$Class == "4"] <- "10"

# If the class is 2 the data will be replaced with 1.
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" & BreastCancer$Class == "2"] <- "1"


#Checking the summary again to see everything is same or if it has changed.
summary(BreastCancer)

# Making sure the data is still classified as factor before checking it.
BreastCancer$Bare_Nuclei <- as.factor(BreastCancer$Bare_Nuclei)

# An alternative to Summary, it displays the class type of each variable along with top 10 instances of data.
str(BreastCancer)






#------------------- Data Slicing, Testing and Training ----------------------

# Data slicing is a step to split data into train and test set.
set.seed(483)

# The method createDataPartition() is used for partitioning our data into train and test set.
# The “Y” parameter takes the value of variable according to which data needs to be partitioned.
# In my case, target variable is the Class Variable.
# P shows the percentage of the split which should stand for 70%.
BreastCancerInTrain <-  createDataPartition(y = BreastCancer$Class, p = 0.7, list = FALSE)

Training <- BreastCancer[BreastCancerInTrain,]

Testing <- BreastCancer[-BreastCancerInTrain,]

dim(Training)

dim(Testing)

anyNA(BreastCancer)

summary(BreastCancer)

Training[["Class"]] = factor(Training[["Class"]])

TrainControl <-
  trainControl(method = "repeatedcv", number = 10, repeats = 3)









# ----------------- SVM Classifier using Linear Kernel --------------------

      #-------------------- SVM LINEAR CLASSIFIER --------------------


svm_Linear <- train(Class ~.,
              data = Training, method = "svmLinear",
              trControl = TrainControl,
              preProcess = c("center", "scale"),tuneLength = 10)

svm_Linear



TestPrediction <- predict(svm_Linear, newdata = Testing)

TestPrediction

confusionMatrix(table(TestPrediction, Testing$Class))




        #-------------------- SVM LINEAR GRID --------------------

grid <-expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5))

set.seed(483)

svm_Linear_Grid <- train(Class ~.,
                   data = Training, method = "svmLinear", 
                   trControl = TrainControl, preProcess = c("center", "scale"), 
                   tuneGrid = grid, tuneLength = 10)

svm_Linear_Grid

plot(svm_Linear_Grid)



TestPrediction_grid <- predict(svm_Linear_Grid, newdata = Testing)

TestPrediction_grid

confusionMatrix(table(TestPrediction_grid, Testing$Class))







# ----------------- SVM Classifier using Non-Linear Kernel --------------------

      #-------------------- SVM RADIAL CLASSIFIER --------------------



set.seed(483)

svm_Radial <- train(Class ~.,
              data = Training, method = "svmRadial",
              trControl = TrainControl, 
              preProcess = c("center", "scale"), tuneLength = 10)


svm_Radial

plot(svm_Radial)



TestPrediction_Radial <- predict(svm_Radial, newdata = Testing)

TestPrediction_Radial

confusionMatrix(table(TestPrediction_Radial, Testing$Class))




      #-------------------- SVM RADIAL GRID --------------------

grid_Radial <-
  expand.grid(
    sigma = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5),
    C =     c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5)
  )
set.seed(483)

svm_Radial_Grid <- train(Class ~.,
                   data = Training, method = "svmRadial",
                   trControl = TrainControl, preProcess = c("center", "scale"),
                   tuneGrid = grid_Radial, tuneLength = 10)

svm_Radial_Grid

plot(svm_Radial_Grid)



TestPrediction_Radial_Grid <- predict(svm_Radial_Grid, newdata = Testing)

TestPrediction_Radial_Grid

confusionMatrix(table(TestPrediction_Radial_Grid, Testing$Class))
