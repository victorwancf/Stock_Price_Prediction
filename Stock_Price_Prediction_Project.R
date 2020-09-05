if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library (readr)
library(randomForest)

#Download The Data Set
# Facebook Stock Price dataset:
# https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv
# Nasdaq Daily Index dataset:
# https://github.com/victorwancf/Stock_Price_Prediction/raw/master/Nasdaq.csv

urlfile="https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv"
FB <- read_csv(url(urlfile))
head(FB)

urlfile2="https://github.com/victorwancf/Stock_Price_Prediction/raw/master/Nasdaq.csv"
nasdaq <- read_csv(url(urlfile2))
head(nasdaq)

#Dataset Structure
str(FB)
str(nasdaq)

#Data Cleaning
colnames(FB) <- c("Date", "Open", "High","Low","Close","Adjusted_Close","Volume")
FB <- as.data.frame(FB) %>% mutate(Date = as.Date(Date),
                                   Open = as.numeric(Open),
                                   High = as.numeric(High),
                                   Low = as.numeric(Low),
                                   Close = as.numeric(Close),
                                   Adjusted_Close = as.numeric(Adjusted_Close),
                                   Volume = as.numeric(Volume))
sum(is.na(FB))
nrow(FB)

colnames(nasdaq) <- c("Date", "N_Open", "N_High","N_Low","N_Close","N_Adjusted_Close","N_Volume")
nasdaq <- as.data.frame(nasdaq) %>% mutate(Date = as.Date(Date),
                                   N_Open = as.numeric(N_Open),
                                   N_High = as.numeric(N_High),
                                   N_Low = as.numeric(N_Low),
                                   N_Close = as.numeric(N_Close),
                                   N_Adjusted_Close = as.numeric(N_Adjusted_Close),
                                   N_Volume = as.numeric(N_Volume))
sum(is.na(nasdaq))
nrow(nasdaq)

#Data Wrangling
stock <- left_join(FB, nasdaq ,by = "Date")
# stock %>% ggplot(aes(Date)) +
#   geom_line(aes(y = Adjusted_Close), colour = "red") +
#   geom_line(aes(y = N_Adjusted_Close), colour = "lightblue") +
#   ggtitle("Adjusted Close Price of Facebook and NASDAQ from 2012 to 2020")

#Plotting of Adjusted_Close of Facebook
stock %>% ggplot(aes(Date,Adjusted_Close)) +
  geom_line(colour = "lightblue", size = 1) +
  ggtitle("Adjusted Close Price of Facebook from 2012 to 2020")

#Adjusted_Close Variation
stock %>% mutate(Adj_Close_pct = (Adjusted_Close/lag(Adjusted_Close)-1)*100) %>%
  ggplot(aes(Adj_Close_pct)) +
  geom_histogram(bins = 50,fill = "lightblue") +
  ggtitle("Percentage change of Daily Adjusted Close Price")


#Seperating Trainning set, Test set and Validation Set
#Ratio of Train : Test : Valid = 6:2:2
stock_Train_set <- stock[1:1245,]
stock_Test_set <- stock[1246:1660,]
stock_Validation <- stock[1661:2076,]

#Dataset: 2012-05-18 to 2020-08-18
#Trainset: 2012-05-18 to 2017-05-01
#Testset: 2017-05-01 to 2018-12-21
#Validation: 2018-12-24 to 2020-08-18

#Cleaning unnecessary variables
rm(urlfile)

#Define RMSE
RMSE <- function(true_value, predicted_value){
  sqrt(mean((true_value - predicted_value)^2))
}

#Model Building
#Multivariate Regression
plot(stock$Adjusted_Close,stock$Open)
plot(stock$Adjusted_Close,stock$N_Open)
mr <- lm(Adjusted_Close ~ Open + N_Open, data = stock_Train_set)
summary(mr)

stock_pred_mr <- predict(mr,stock_Test_set)
Multi_rmse <- RMSE(stock_Test_set$Adjusted_Close,stock_pred_mr)
rmse_results <- tibble(method = "Multivariate Regression", RMSE = Multi_rmse)
rmse_results %>% knitr::kable()

stock_mr <- cbind(stock_Test_set,stock_pred_mr) %>%
  mutate(Adjusted_Close = stock_pred_mr,
         dataset = "Multivariate Regression")

#Plotting Multivariate Regression Result
stock_train <- stock_Train_set %>% mutate(dataset = "train set")
stock_test <- stock_Test_set %>% mutate(dataset = "test set")
bind_rows(stock_train,stock_test,stock_mr) %>% 
  ggplot(aes(Date,Adjusted_Close,col = dataset)) +
  geom_line()

###################################################################
#KNN
control <- trainControl(method = "cv", number = 5, p = .9)
train_knn <- train(stock_Train_set[2], stock_Train_set$Adjusted_Close, 
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(3,21,2)),
                   trControl = control)

train_knn

fit_knn <- knn3(stock_Train_set[2], as.factor(stock_Train_set$Adjusted_Close),  k = 15)
stock_pred_knn <- predict(fit_knn, stock_Test_set[2], type="class")
knn_rmse <- RMSE(as.numeric(stock_pred_knn),stock_Test_set$Adjusted_Close)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="KNN",
                                     RMSE = knn_rmse))
rmse_results %>% knitr::kable()

#Plotting of Result (KNN)
stock_knn <- cbind(stock_Test_set,stock_pred_knn) %>%
  mutate(Adjusted_Close = as.numeric(stock_pred_knn),
         dataset = "KNN")

bind_rows(stock_train,stock_test,stock_knn) %>% 
  ggplot(aes(Date,Adjusted_Close,col = dataset)) +
  geom_line()

###################################################################
# Random Forest
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100, 200))

stock_train_rf <-  train(stock_Train_set[c(2,8)], stock_Train_set$Adjusted_Close, 
                   method = "rf", 
                   ntree = 300,
                   trControl = control,
                   tuneGrid = grid)
ggplot(stock_train_rf)
stock_train_rf$bestTune

stock_fit_rf <- randomForest(stock_Train_set[c(2,8)], stock_Train_set$Adjusted_Close, 
                       minNode = stock_train_rf$bestTune$mtry)

plot(stock_fit_rf) #Check to see if we ran enough tree

stock_pred_rf <- predict(stock_fit_rf, stock_Test_set)
rf_rmse <- RMSE(as.numeric(stock_pred_rf),stock_Test_set$Adjusted_Close)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Random Forest",
                                     RMSE = rf_rmse))
rmse_results %>% knitr::kable()

#Plotting of Result (Random Forest)
stock_rf <- cbind(stock_Test_set,stock_pred_rf) %>%
  mutate(Adjusted_Close = stock_pred_rf,
         dataset = "Random Forest")

#Plotting test Set and Predicted


#Plotting all in one
bind_rows(stock_train,stock_test,stock_rf) %>% 
  ggplot(aes(Date,Adjusted_Close,col = dataset))  +
  geom_line()


#Model Evaluation

rmse_results %>% knitr::kable()
