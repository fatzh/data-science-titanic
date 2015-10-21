# this is the external script for the exploration Rmarkdown document

## ---- loadPackages --------------
library('ggplot2')
library('dplyr')
library('tidyr')
library('reshape2')
library('scales')

## ---- loadData
df <- read.csv('./predictions/baseline_svm_predictions.csv', sep=',')

## ---- splitDataSet
# to try: group using a factor, and show only the more important features.
df$group[df$Survived == df$Predicted & df$Predicted == 1] <- "True positive"
df$group[df$Survived == df$Predicted & df$Predicted == 0] <- "True negative"
df$group[df$Survived < df$Predicted] <- "False positive"
df$group[df$Survived > df$Predicted] <- "False negative"

df$group <- as.factor(df$group)
df$group = factor(df$group, levels(df$group)[c(3,1,4,2)])

## ---- scale
df$cabin_count = rescale(df$cabin_count, to = c(0,1), from = range(df$cabin_count))
df$family_size = rescale(df$family_size, to = c(0,1), from = range(df$family_size))
df$fare = rescale(df$fare, to = c(0,1), from = range(df$fare))
df$age = rescale(df$age, to = c(0,1), from = range(df$age))

## ---- plotMeans
d_means <- df %>% 
    select(-Predicted, -Survived) %>% 
    group_by(group) %>% 
    summarise_each(funs(mean)) %>% 
    melt(id.vars=c('group')) %>%
    rename(feature=variable)

g_means <- ggplot(data=d_means, aes(x=group, y=feature)) + 
    geom_tile(aes(fill=value), colour="white") + 
    scale_fill_gradient(low="white", high="steelblue") +
    theme(axis.ticks = element_blank(), axis.text.y = element_blank()) + scale_y_discrete(breaks=NULL)

## ---- plotMeansImportant
d_important <- df %>% 
    select(-Predicted, -Survived) %>%
    select(age, 
           fare, 
           title_Mr, 
           sex, 
           title_Miss, 
           family_size, 
           bracket, 
           title_Mrs, 
           class_3, 
           cabin_count, 
           first_ticket_digit_1, 
           quotes, 
           class_1, 
           class_2, 
           first_ticket_digit_3, 
           port_S, 
           first_ticket_digit_2, 
           port_C, 
           title_Sir, 
           firstname_Anna, 
           group) %>% 
    group_by(group) %>% 
    summarise_each(funs(mean)) %>% 
    melt(id.vars=c('group')) %>%
    mutate(variable=factor(variable, rev(levels(variable)))) %>%
    rename(feature=variable)

g_important <- ggplot(data=d_important, aes(x=group, y=feature)) + 
    geom_tile(aes(fill=value), colour="white") + 
    scale_fill_gradient(low="white", high="steelblue") +
    theme(axis.text.y = element_text(angle = 45, hjust=1))