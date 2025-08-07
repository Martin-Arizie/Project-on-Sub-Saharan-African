#  Author ----
#  Martin Arizie
#  Date Updated ----
#  2025-07-24

library(DiagrammeR)
library(Matrix)
library(readxl)
library(tidyverse)
library(janitor)
library(stringr)
library(Metrics)
library(lubridate)
library(haven)
library(here)
library(gt)
library(tibble)
library(plm)
library(stargazer)
library(caret)
library(tidyverse)
library(dplyr)
library(caret)
library(xtsum)
library(keras)
library(tensorflow)
library(Metrics)
library(MASS)
library(neuralnet)
library(e1071)
library(rnn)
library(RSNNS)
library(ggplot2)
library(xgboost)
library(nnet)
library(iml)
library(earth)
library(vip)
library(NeuralNetTools)
library(mlr)

dn <- read_xlsx("PanelData.xlsx")

data<- dn |> 
  mutate(
    GDP = as.double(GDP),
    GDP =replace_na(GDP, median(GDP,na.rm = T)),
    Hydroelectric = as.double(Hydroelectric ),
    Hydroelectric =replace_na(Hydroelectric, median(Hydroelectric,na.rm = T)),
    Naturalgas = as.double(Naturalgas),
    Naturalgas=replace_na(Naturalgas, median(Naturalgas,na.rm = T)),
    OGC= as.double(OGC),
    OGC =replace_na(OGC, median(OGC,na.rm = T))
  )

view(data)

maa<-data[-c(1:2)]

view(maa)
############################################################################
########## Building a Multilayer Perceptrone ###############################
#########################################################################

# Splitting Data into Training (70%) and Testing (30%)
set.seed(123)
normalize1 <- function(x) (x - min(x)) / (max(x) - min(x))
data[, 3:6] <- as.data.frame(lapply(data[, 3:6], normalize1))
part<-createDataPartition(data$GDP, p=0.70, list=F)
train.set<-data[part,]
train.set<-train.set[-1]
testing.set<-data[-part,]
testing.set<-testing.set[-1]

# Define Input (X) and Output (Y)
dtrain <- as.matrix( train.set[, c("Hydroelectric","Naturalgas","OGC")])
ydtrain <- as.matrix( train.set[, "GDP"])   
dtest <- as.matrix( testing.set[, c("Hydroelectric","Naturalgas","OGC")])
ydtest <- as.matrix( testing.set[, "GDP"])
view(dtest)
###Training Multilayer Perceptrone Model####


model<-mlp( dtrain, ydtrain,
            size = c(5,2),
            maxit = 150,
            learningrate=0.05,
            initFunc = "Randomize_Weights",
            initFuncParams = c(-0.5, 0.5),
            learnFunc = "Std_Backpropagation",
            learnFuncParams = c(0.1, 0),
            updateFunc = "Topological_Order",
            updateFuncParams = c(0),
            hiddenActFunc = "Act_Logistic",
            shufflePatterns = TRUE,
            linOut = TRUE,
            validation_split=0.5
)

summary(model)


###### MAKING PREDICTION USING THE TRAIN DATASET ########

pred_train_y <-  predict(model, dtrain)
print(pred_train_y)

####COR ###
Train_Cor<-cor(ydtrain, pred_train_y )
Train_Cor

#### MEAN ABSOLUTE PERCENTAGE ERROR OF MLP MODEL USING TRAIN DATASET ###
Train_MAPE<-mean(abs(ydtrain-pred_train_y))*100
Train_MAPE
#### MEAN SQUARE ERROR OF MLP MODEL USING TRAIN DATASET ###
Train_MSE<-mean( ydtrain-pred_train_y)^2
Train_MSE

### ROOT MEAN SQUARE ERROR OF MLP MODEL USING TRAIN DATASET###
#Train_RMSE<-sqrt(mean(ydtrain-pred_train_y)^2)
#Train_RMSE

#############Test Set ###################
pred_test_y <- predict(model, dtest)
pred_test_y

###Cor ############
Test_cor<-cor(ydtest, pred_test_y )
Test_cor

#### MEAN ABSOLUTE PERCENTAGE ERROR OF MLP MODEL USING TEST DATASET ###
Test_MAPE<-mean(abs(ydtest-pred_test_y ))*100
Test_MAPE
#### MEAN SQUARE ERROR OF MLP MODEL USING TEST DATASET ###
Test_MSE<-mean((ydtest-pred_test_y )^2)
Test_MSE
### ROOT MEAN SQUARE ERROR OF MLP MODEL USING TEST DATASET###
#Test_RMSE<-sqrt(mean(ydtest-pred_test_y )^2)
#Test_RMSE




######Variable Importance################
windows(height=10,width=20)
importance <- olden(model)
# Print the importance values
print(importance)

#####Extract the biasis and other relevant information#######

extractNetInfo(model)

### Visualizing the training error by iteration ###
windows(height=10,width=20)
plotIterativeError(model)


##Epoch 200 ##
model2<-mlp( dtrain, ydtrain,
            size = c(5,2),
            maxit = 200,
            learningrate=0.01,
            initFunc = "Randomize_Weights",
            initFuncParams = c(-0.5, 0.5),
            learnFunc = "Std_Backpropagation",
            learnFuncParams = c(0.1, 0),
            updateFunc = "Topological_Order",
            updateFuncParams = c(0),
            hiddenActFunc = "Act_Logistic",
            shufflePatterns = TRUE,
            linOut = TRUE,
            validation_split=0.2
)

summary(model2)
#####Extract the biasis and other relevant information#######

extractNetInfo(model2)

### Visualizing the training error by iteration ###
windows(height=10,width=20)
plotIterativeError(model2)

 ###########################################
##########B PART #################

# Ensure country column exists
if (!"Country" %in% colnames(data)) {
  stop("The dataset must contain a 'Country' column.")
}

# Initialize results dataframe
results <- data.frame(Country = character(),
                      Train_MSE = numeric(), Test_MSE = numeric(), Train_MAPE = numeric(), Test_MAPE = numeric())

# Process each country separately
for (country in unique(data$Country)) {
  cat("Processing:", country, "\n")
  
  country_data <- filter(data, Country == country)
  
  # Ensure enough data points exist
  if (nrow(country_data) < 5) next
  
  # Train-test split
  set.seed(123)
  trainIndex <- createDataPartition(country_data$GDP, p = 0.70, list = FALSE)
  trainData <- country_data[trainIndex, ]
  testData <- country_data[-trainIndex, ]
  
  # Prepare X and Y matrices
  X_train <- as.matrix(trainData[,-c(1,2)])
  y_train <- trainData$GDP
  X_test <- as.matrix(testData[,-c(1,2)])
  y_test <- testData$GDP
  
  
  
  # Define Input (X) and Output (Y)
  dtrain <- as.matrix( train.set[, c("Hydroelectric","Naturalgas","OGC")])
  ydtrain <- as.matrix( train.set[, "GDP"])   
  dtest <- as.matrix( testing.set[, c("Hydroelectric","Naturalgas","OGC")])
  ydtest <- as.matrix( testing.set[, "GDP"])
  
  
  model<-mlp( dtrain, ydtrain,
              size = c(5,3),
              maxit = 100,
              initFunc = "Randomize_Weights",
              initFuncParams = c(-0.3, 0.3),
              learnFunc = "Std_Backpropagation",
              learnFuncParams = c(0.2, 0),
              updateFunc = "Topological_Order",
              updateFuncParams = c(0),
              hiddenActFunc = "Act_Logistic",
              shufflePatterns = TRUE,
              linOut = TRUE,
              validation_split=0.2
  )
  
  # Predictions
  y_train_pred <- predict(model, dtrain)
  y_test_pred <- predict(model, dtest)
  
  # Performance metrics
  #train_cor <- cor(y_train, y_train_pred, use = "complete.obs")
  #test_cor <- cor(y_test, y_test_pred, use = "complete.obs")
  train_mse <- mean((y_train - y_train_pred)^2)
  test_mse <- mean((y_test - y_test_pred)^2)
  train_mape <- mean(abs((y_train - y_train_pred) / y_train))
  test_mape <- mean(abs((y_test - y_test_pred) / y_test))
  
  # Store results
  results <- rbind(results, data.frame(Country = country,
                                       Train_MSE = train_mse, Test_MSE = test_mse, Train_MAPE = train_mape, Test_MAPE = test_mape))
}

# Print final results table
print(results)

r_squaredd=1-(sum((ydtest-pred_test_y)^2)/ sum((ydtest-mean(ydtest))^2))


r_squaredd

