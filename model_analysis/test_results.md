---
title: "F1 analysis"
author: "Fabrice Tereszkiewciz"
date: "14 Oct 2015"
output: html_document
---





We can load the data into R and make sure we have the right dimensions.


```r
df <- read.csv('./predictions/baseline_predictions.csv', sep=',')
dim(df)
```

```
## [1] 223 281
```

That's good, we have all our features and the survival status as well as the predicted survival status.

We can also easily see the confusion matrix for this dataset.


```r
table(df$Survived, df$Predicted)
```

```
##    
##       0   1
##   0 124  15
##   1  18  66
```

Here we have the prediction as columns and the truth as rows. This tables tells us that:

- the model has correctly classified 124 victims and 66 survivors
- the model has predicted 15 survivors who actually died
- the model has missed 18 survivors

We can split our 3 groups of interests and see if we can extract some patterns.


```r
# to try: group using a factor, and show only the more important features.
df$group[df$Survived == df$Predicted & df$Predicted == 1] <- "True positive"
df$group[df$Survived == df$Predicted & df$Predicted == 0] <- "True negative"
df$group[df$Survived < df$Predicted] <- "False positive"
df$group[df$Survived > df$Predicted] <- "False negative"

df$group <- as.factor(df$group)
df$group = factor(df$group, levels(df$group)[c(3,1,4,2)])
```

An easy way is to plot the means of each variables for each group. As most of the data are boolean, the means are mostly going ot be between 0 and 1. Let's rescale the other features to fit between 0 and 1, based on the range on the whole dataset.


```r
df$cabin_count = rescale(df$cabin_count, to = c(0,1), from = range(df$cabin_count))
df$family_size = rescale(df$family_size, to = c(0,1), from = range(df$family_size))
df$fare = rescale(df$fare, to = c(0,1), from = range(df$fare))
df$age = rescale(df$age, to = c(0,1), from = range(df$age))
```

And plot the means of each variables.


```r
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
g_means
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 

I ommited the feature names from this plot, as there are too many to read anyway, but the plot itself tells that the dataset seem fairly well balanced, it's not like if we would see a huge difference in one of the four categories.

We can further investigate by looking only at the most important features (we extracted the list previously).


```r
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
g_important
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png) 
