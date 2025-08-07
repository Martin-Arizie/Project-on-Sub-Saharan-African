#  Author ----
#  Martin Arizie
#  Date Updated ----
#  2025-07-23

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
library(factoextra)
library(ggpubr)
library(survminer)
library(ggcorrplot)
library(fastqcr)
library(GGally)

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


###Preliminary Analysis to Determine the Relationship of between the variables##

ma<-data[-c(1:2)]
view(ma)

windows(height=10,width=20)
ggpairs(ma)

############################################################################
########## Building an XGBoost Model ###############################
#########################################################################

# Splitting Data into Training (70%) and Testing (30%)
set.seed(123)
normalize1 <- function(x) (x - min(x)) / (max(x) - min(x))
data[, 3:6] <- as.data.frame(lapply(data[, 3:6], normalize1))
part<-createDataPartition(data$GDP, p=0.7, list=F)
train.set<-data[part,]
train.set<-train.set[-1]
testing.set<-data[-part,]
testing.set<-testing.set[-1]

####Sparsing data for XGboost#####
sparse_xx.tr<-sparse.model.matrix(GDP~Hydroelectric+Naturalgas+OGC, data=train.set)
structure(sparse_xx.tr)
sparse_xx.tee<-sparse.model.matrix(GDP~Hydroelectric+Naturalgas+OGC, data=testing.set)
head(sparse_xx.tee)

####Fit XGBoost Model#####
xgboost_reg<-xgboost(data=sparse_xx.tr, label=train.set$GDP, objective="reg:gamma", nrounds=200, verbose=1)
xgboost_reg



###Variable Importance#####3
importance_matrix<-xgb.importance(model=xgboost_reg)
print(importance_matrix)
windows(height=10,width=20)
xgb.plot.importance(importance_matrix=importance_matrix, col="red")


###it
y_train_pred<-predict(xgboost_reg,sparse_xx.tr, outputmargin = F)

train_mse<-(mean(y_train-y_train_pred)^2)

errors<-xgboost_reg$evaluation_log
errors

windows(height=10,width=20)
ggplot(errors, aes(x=iter, y=train_mse))+
  geom_line()+
  labs(x="Iteration", y="mse")



###################################################
#######General RNN Performance Metrics#############
###################################################

# Prepare X and Y matrices
X_train <- as.matrix(train.set[,-c(1,2)])
y_train <- train.set$GDP
X_test <- as.matrix(testing.set[,-c(1,2)])
y_test <- testing.set$GDP


######## Compute performance metrics of Model #######

#Predicting the output with testing data set#
y_train_pred<-predict(xgboost_reg,sparse_xx.tr, outputmargin = F)
y_train_pred

train_cor<-cor(y_train, y_train_pred)
train_cor

train_mse <- mean((y_train - y_train_pred)^2)
train_mse

#train_rmse<-sqrt(mean(y_train-y_train_pred)^2)
#train_rmse

train_mape <- mean(abs(y_train - y_train_pred))*100
train_mape

###Predicting the output with testing data set####

y_test_pred<-predict(xgboost_reg,sparse_xx.tee, outputmargin = F)
y_test_pred

test_cor<-cor(y_test, y_test_pred)
test_cor

test_mse <- mean((y_test - y_test_pred)^2)
test_mse

#test_rmse<-sqrt(mean(y_test-y_test_pred)^2)
#test_rmse

test_mape <- mean(abs(y_test - y_test_pred))*100
test_mape


################################################################
######### B PART OF XGBoost: Getting the country base mse ######
################################################################

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
  trainIndex <- createDataPartition(country_data$UMP, p = 0.8, list = FALSE)
  trainData <- country_data[trainIndex, ]
  testData <- country_data[-trainIndex, ]
  
  # Prepare X and Y matrices
  X_train <- as.matrix(trainData[,-c(1,2,3)])
  X_train
  y_train <- trainData$UMP
  y_train
  X_test <- as.matrix(testData[,-c(1,2,3)])
  X_test
  y_test <- testData$UMP
  y_test
  
  ####Sparsing data for XGboost#####
  dtrain<-sparse.model.matrix(UMP~CAB+ER+FDI+PopG+General_GFCE+Personal_RR+Trade+Inflation+AFF+FCE+Gross_savings+FDI_IN, data=trainData)
  structure(dtrain)
  dtest <-sparse.model.matrix(UMP~CAB+ER+FDI+PopG+General_GFCE+Personal_RR+Trade+Inflation+AFF+FCE+Gross_savings+FDI_IN, data=testData)
  head(dtest )
  
  model<-xgboost(data= dtrain, label=trainData$UMP, objective="reg:gamma", nrounds=100, verbose=1)
  
  # Predictions
  y_train_pred <- predict(model, dtrain)
  y_test_pred <- predict(model, dtest)
  
  # Performance metrics
  #train_cor <- cor(y_train, y_train_pred, use = "complete.obs")
  #test_cor <- cor(y_test, y_test_pred, use = "complete.obs")
  
  train_mse <- mean((y_train - y_train_pred)^2)
  test_mse <- mean((y_test - y_test_pred)^2)
 # train_rmse<-sqrt(mean(y_train-y_train_pred)^2)
 # test_rmse<-sqrt(mean(y_test-y_test_pred)^2)
  train_mape <- mean(abs((y_train - y_train_pred) / y_train))
  test_mape <- mean(abs((y_test - y_test_pred) / y_test))
  # Store results
  results <- rbind(results, data.frame(Country = country, Train_MSE = train_mse,Test_MSE = test_mse, 
            Train_MAPE = train_mape, Test_MAPE = test_mape))
}

# Print final results table
print(results)
