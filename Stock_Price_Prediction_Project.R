if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(TTR)) install.packages("TTR", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library (readr)
library(rpart)
library(randomForest)
library(TTR)

#Download The Data Set
# Facebook Stock Price dataset:
# https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv
# Nasdaq Daily Index dataset:
# https://github.com/victorwancf/Stock_Price_Prediction/raw/master/Nasdaq.csv

urlfile="https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv"
stock <- read_csv(url(urlfile))

#Rename columns and change the class of data
colnames(stock) <- c("Date", "Open", "High","Low","Close","Adjusted_Close","Volume")
stock <- as.data.frame(stock) %>% mutate(Date = as.Date(Date),
                                         Open = as.numeric(Open),
                                         High = as.numeric(High),
                                         Low = as.numeric(Low),
                                         Close = as.numeric(Close),
                                         Adjusted_Close = as.numeric(Adjusted_Close),
                                         Volume = as.numeric(Volume))

head(stock)

#Visualization of Facebook stock price
stock %>% ggplot(aes(Date,Adjusted_Close)) +
  geom_line(colour = "lightblue", size = 1) +
  ggtitle("Adjusted Close Price of Facebook from 2012 to 2020")

#Adding indicators
stock <- stock %>% mutate(Change_in_price = Adjusted_Close - lag(Adjusted_Close),
                          Singal_Flag = as.factor(sign(Change_in_price)),
                          RSI = RSI(Adjusted_Close,n=14),
                          MACD = MACD(Adjusted_Close, 12, 26, 9)[,"macd"],
                          WPR = WPR(stock[,c("High","Low","Close")]),
                          stoch = stoch(stock[,c("High","Low","Close")],14,3,3)[,"fastK"],
                          ROC = ROC(Adjusted_Close, n = 9),
                          OBV = OBV(Adjusted_Close,Volume))

# A standard window is used for all indicators, such as RSI(14), MACD(12,26,9), Stoch(14,3,3) and ROC(9).

#Remove NAs
stock <- na.omit(stock)

#Create train,test and validation set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = stock$Singal_Flag, times = 1, p = 0.1, list = FALSE)
edx <- stock[-test_index,] #edx set is used for seperating Train set and the Test set
stock_Validation <- stock[test_index,]
test_index <- createDataPartition(stock$Singal_Flag, times = 1, p = 0.2, list = FALSE)
stock_Train_set <- edx %>% slice(-test_index)
stock_Test_set <- edx %>% slice(test_index)


#KNN model
set.seed(1)
control <- trainControl(method = "cv", number = 4, p = .9)
train_knn <- train(Singal_Flag ~ RSI + OBV + WPR + stoch + Open +  MACD + ROC , data = stock_Train_set, 
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(3,21,2)),
                   trControl = control)

plot(train_knn)

fit_knn <- knn3(Singal_Flag ~ RSI + OBV + WPR + stoch + Open +  MACD + ROC  , data = stock_Train_set,  k = 5)
stock_pred_knn <- predict(fit_knn, stock_Test_set, type="class")

knn_acc <- confusionMatrix(stock_pred_knn,stock_Test_set$Singal_Flag)$overall["Accuracy"]

acc_results <- tibble(method = "KNN", Accuracy = knn_acc)
acc_results %>% knitr::kable()

#Random forest model
set.seed(1)
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100, 200))

stock_train_rf <-  train(Singal_Flag ~ RSI + ROC + OBV + WPR + ROC , data = stock_Train_set,
                         method = "rf", 
                         ntree = 300,
                         trControl = control,
                         tuneGrid = grid)

stock_fit_rf <- randomForest(Singal_Flag ~ RSI + OBV + WPR + stoch + Open +  MACD + ROC,  data = stock_Train_set,
                             minNode = stock_train_rf$bestTune$mtry)

stock_pred_rf <- predict(stock_fit_rf, stock_Test_set)

RF_acc <- confusionMatrix(stock_pred_rf,stock_Test_set$Singal_Flag)$overall["Accuracy"]
acc_results <- bind_rows(acc_results,
                         data_frame(method="Random Forest",
                                    Accuracy = RF_acc))
acc_results %>% knitr::kable()

#Results in validation set
set.seed(1)
stock_fit_rf <- randomForest(Singal_Flag ~ RSI + OBV + WPR + stoch + Open +  MACD + ROC,  data = stock_Train_set,
                             minNode = stock_train_rf$bestTune$mtry)

stock_pred_rf <- predict(stock_fit_rf, stock_Validation)

RF_acc <- confusionMatrix(stock_pred_rf,stock_Validation$Singal_Flag)$overall["Accuracy"]
acc_results <- data_frame(method="Random Forest on validation set",
                          Accuracy = RF_acc)
acc_results %>% knitr::kable()

#Plotting of varibale importance
varImpPlot(stock_fit_rf, main = "Plot of variable importance in Random Forest")
