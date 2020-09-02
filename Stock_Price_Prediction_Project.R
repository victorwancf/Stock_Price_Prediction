if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library (readr)

#Download The Data Set
# Facebook Stock Price dataset:
# https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv

urlfile="https://github.com/victorwancf/Stock_Price_Prediction/raw/master/FB.csv"
FB <-read_csv(url(urlfile))
head(FB)

#Data Exploration
str(FB)

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

#Data Wrangling

#Plotting of Close Price of Facebook
FB %>% ggplot(aes(Date,Close)) +
  geom_line(colour = "lightblue", size = 1) +
  ggtitle("Close Price of Facebook from 2012 to 2020")
  

#Seperating Trainning set, Test set and Validation Set
#Ratio of Train : Test : Valid = 6:2:2
FB_Train_set <- FB[1:1245,]
FB_Test_set <- FB[1246:1660,]
FB_Validation <- FB[1661:2076,]

#Dataset: 2012-05-18 to 2020-08-18
#Trainset: 2012-05-18 to 2017-05-01
#Testset: 2017-05-01 to 2018-12-21
#Validation: 2018-12-24 to 2020-08-18

# rm(urlfile)

#Define RMSE
RMSE <- function(true_value, predicted_value){
  sqrt(mean((true_value - predicted_value)^2))
}

#Model Building
#Linear Regression


#KNN
