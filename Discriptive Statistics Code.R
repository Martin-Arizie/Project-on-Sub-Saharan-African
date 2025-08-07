# Project Title: Analysis of thesis data
#  Author ----
#  Martin
#  Date Updated ----
#  2025-7-25

library(readxl)
library(tidyverse)
library(janitor)
library(stringr)
library(Metrics)
library(lubridate)
library(haven)
library(here)
library(tibble)
library(plm)
library(stargazer)
library(caret)
library(tidyverse)
library(dplyr)
library(caret)
library(xtsum)
library(psych)
library(keras)
library(tensorflow)
library(MASS)
library(neuralnet)
library(e1071)
library(rnn)
library(RSNNS)
library(ggplot2)
library(VGAM)
library(AER)

dn <- read_xlsx("PanelDiscrip.xlsx")

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
# Summary Statistics
# for panel data 
xtsum::xtsum(data, id = "ID", t = "Year", na.rm = T, dec = 4)



##FINDING KURTOSIS AND SKEWNESS##
#CHANGING TO LONGER FORMAT#
d3 <- data |> 
  pivot_longer(cols = c(GDP,Hydroelectric,Naturalgas,OGC),
               names_to = "Inc")


# by Variables
kurskw <- d3 |> group_by(Inc)|> 
  summarize(
    kurtosis = kurtosis(value, type=1 ),
    skewness = skewness(value, type = 1)
  )
print(kurskw) 
