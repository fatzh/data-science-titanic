Titanic: Machine Learning from Disaster
=======================================

Introduction
------------

The Kaggle competition "Titanic: Machine Learning from Disaster" offers
the opportunity to analyse the outcome of the Titanic disaster. We have
access to the list of passengers who survived or not along with
information regarding their situation (passenger class, travelled alone
or in family, etc..).

The goal is to build a model to predict if a passenger survived based on
this data. We will first look at the raw data to get a better picture of
the parameters that are more or less correlated with the survival of the
passenger. Also we'll study the different features in order to determine
if more information can be obtained by derivation.

Loading data
------------

The data is in CSV format. You must download it from the Kaggle
competition and place it in a `data` folder. I also load the test data,
may be handy if we decide some transformation.

    df <- read.csv2('../data/train.csv', sep = ',')
    df_test <- read.csv2('../data/test.csv', sep=',')

A quick look at the data:

    glimpse(df)

    ## Observations: 891
    ## Variables: 12
    ## $ PassengerId (int) 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,...
    ## $ Survived    (int) 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0,...
    ## $ Pclass      (int) 3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3,...
    ## $ Name        (fctr) Braund, Mr. Owen Harris, Cumings, Mrs. John Bradl...
    ## $ Sex         (fctr) male, female, female, female, male, male, male, m...
    ## $ Age         (fctr) 22, 38, 26, 35, 35, , 54, 2, 27, 14, 4, 58, 20, 3...
    ## $ SibSp       (int) 1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4,...
    ## $ Parch       (int) 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1,...
    ## $ Ticket      (fctr) A/5 21171, PC 17599, STON/O2. 3101282, 113803, 37...
    ## $ Fare        (fctr) 7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625...
    ## $ Cabin       (fctr) , C85, , C123, , , E46, , , , G6, C103, , , , , ,...
    ## $ Embarked    (fctr) S, C, S, S, S, Q, S, S, S, C, S, S, S, S, S, S, Q...

We have 891 observations, 12 features.

Features analysis
-----------------

### Passenger Id

This is obviously not useful to determine if the passenger survived or
not. We will exclude it in the preprocessing stage.

### Survived

This is the outcome of the prediction, we will use it to train our
model. This shouldn't be an `integer` but a factor. Let's see how many
passengers in our dataset survived the disaster:

    df$Survived <- as.factor(df$Survived)
    survived <- count(df[df$Survived == 1,])$n
    died <- count(df[df$Survived == 0,])$n
    survival_rate <- round(survived / (survived + died) *100, 2)
    d <- data.frame(
        x = factor(c('Survived', 'Died')),
        y = c(count(df[df$Survived == 1,])$n, count(df[df$Survived == 0,])$n)
    )
    d

    ##          x   y
    ## 1 Survived 342
    ## 2     Died 549

    g <- ggplot(d, aes(x=x, y=y))
    g <- g + geom_bar(stat='identity', aes(fill=x))
    g <- g + labs(title="Titanic survivors", x='', y='')
    g

![](/sites/default/files/basic_page/titanic-4-1.png)

We have 342 survivors and 549 passengers who didn't make it. That's a
38.38% survivial rate.

### Passenger class

That's the feature `Pclass` in the dataset. This can be considered a
proxy for the social status of the passenger (i.e. 1st class passenger
are probably wealthier than 3rd class passengers).

    unique(df$Pclass)

    ## [1] 3 1 2

We have only 3 classes. Let's make a factor out of it and look at the
repartition.

    df$Pclass <- as.factor(df$Pclass)
    d <- data.frame(
        class = factor(c('1st class', '2nd class', '3rd class')),
        passengers = c(count(df[df$Pclass == 1,])$n,
              count(df[df$Pclass == 2,])$n,
              count(df[df$Pclass == 3,])$n)
    )
    g <- ggplot(d, aes(x=class, y=passengers))
    g <- g + geom_bar(stat='identity', aes(fill=class))
    g <- g + labs(title="Passenger class population", x='', y='')
    d

    ##       class passengers
    ## 1 1st class        216
    ## 2 2nd class        184
    ## 3 3rd class        491

    g

![](/sites/default/files/basic_page/titanic-6-1.png)

We can see that the majority of the passengers are in the 3rd class.
Let's see among each class, how many passenger survived the disaster.

    d <- df %>% select(Survived, Pclass)
    g <- ggplot(data = d, aes(x=Pclass, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors by passenger class", x="Passenger class", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-7-1.png)
/sites/default/files/basic_page/titanic
We can see that the majority of the passengers in first class survived,
so did about half of those in second class. The majority of the losses
come from the 3rd class.

The passenger class is therefore to be considered an important feature
for our model.

### Passenger name

This feature as it is probably not relevant, but we can still have a
look at a few caracteristics and see if we find some correlation with
the survival rate. This feature shouldn't be treated as a factor but as
a string.

First, some names have an additional name in brackets. That may also be
an indicator of the social status of the passenger. Let's have a look.

    d <- df %>% mutate(AdditionalName = grepl('\\(', Name)) %>% select(Survived, AdditionalName) 
    g <- ggplot(data = d, aes(x=AdditionalName, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors with brackets in their names", x="Brackets in name", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-9-1.png)

Well, it looks like if you have an additional name in brackets, it
indeed increase your probability of surviving from the disaster. That's
a feature we can use in our model.

Also some names have a name in quotes, maybe something to look at.

    d <- df %>% mutate(QuotedName = grepl('\\"', Name)) %>% select(Survived, QuotedName) 
    g <- ggplot(data = d, aes(x=QuotedName, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors with quotes in their names", x="Quotes in name", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-10-1.png)

Indeed, with quotes, the survival rate is higher than without.

Another information from the name could be the title. Let's extract them
first:

    titles <- as.factor(gsub(".*,\\s?\\w*?\\s?(\\w+)\\..*", "\\1", df$Name, perl=TRUE))
    test_titles <- as.factor(gsub(".*,\\s?\\w*?\\s?(\\w+)\\..*", "\\1", df_test$Name, perl=TRUE))
    levels(titles)

    ##  [1] "Capt"     "Col"      "Countess" "Don"      "Dr"       "Jonkheer"
    ##  [7] "Lady"     "Major"    "Master"   "Miss"     "Mlle"     "Mme"     
    ## [13] "Mr"       "Mrs"      "Ms"       "Rev"      "Sir"

    summary(titles)

    ##     Capt      Col Countess      Don       Dr Jonkheer     Lady    Major 
    ##        1        2        1        1        7        1        1        2 
    ##   Master     Miss     Mlle      Mme       Mr      Mrs       Ms      Rev 
    ##       40      182        2        1      517      125        1        6 
    ##      Sir 
    ##        1

Looks like we can do some cleaning, the infered important imformation we
can hope to extract from the titels being the sex (we already know) and
some kind of social marker. So maybe we can group the "fancy" titles
together, and keep sex separation. We can group distinguished male
titles into 'Sir', keep Mr for the others, and sitinguished female
titles into 'Lady', maybe stil keep the 'Mrs' and 'Miss' as well. Also
Wikipedia tells us that 'Jonkheer' is indeed an honoric title for men :
*Jonkheer (female equivalent: Jonkvrouw) is a Dutch honorific of
nobility*

Before cleaning up, it may worth having a look at the titles in the test
data as well.

    summary(test_titles)

    ##    Col   Dona     Dr Master   Miss     Mr    Mrs     Ms    Rev 
    ##      2      1      1     21     78    240     72      1      2

There is a new 'Dona' that we must correctly group as well.

    titles[titles %in% c('Rev', 'Dr', 'Jonkheer', 'Major', 'Master', 'Capt', 'Col', 'Don')] = 'Sir'
    titles[titles %in% c('Countess', 'Dona')] = 'Lady'
    titles[titles %in% c('Mlle', 'Ms')] = 'Miss'
    titles[titles == 'Mme'] = 'Mrs'
    titles = droplevels(titles)
    summary(titles)

    ## Lady Miss   Mr  Mrs  Sir 
    ##    2  185  517  126   61

Here are the survival rates:

    d <- df %>%
        mutate(title = titles) %>%
        select(Survived, title) %>%
        group_by(title, Survived) %>% 
        summarise(n = n()) %>% 
        mutate(rate = n/sum(n) * 100) %>% 
        filter(Survived == 1)
    g <- ggplot(data = d, aes(x=title, y=rate))
    g <- g + geom_bar(stat='identity') 
    g <- g + labs(title="Survival rate by title", x="Title", y="Survival rate in %") 
    g

![](/sites/default/files/basic_page/titanic-14-1.png)

We can see that the survival rate varies by title. `Mr` died the most,
and the \``Lady` survived. So this would also be an interesting feature
to include in our model. Also... with the little test trick, we can hope
that our 'Dona' will survive our model.

We can also look at the firstnames, that may be an indication of the
social status.

    firstname <- as.factor(gsub(".*\\. o?f? ?\\(?(\\w+).*", "\\1", df$Name, perl=TRUE))
    d <- df %>% mutate(firstname = firstname)
    common_firstnames <- d %>% 
        group_by(firstname) %>% 
        summarise(count=n()) %>% 
        arrange(desc(count)) %>% 
        filter(count > 5) %>%
        select(firstname)
    d2 <- d %>%
        select(Survived, firstname) %>%
        group_by(firstname, Survived) %>% 
        filter(firstname %in% common_firstnames$firstname) %>%
        summarise(n = n()) %>% 
        mutate(rate = n/sum(n) * 100) %>% 
        filter(Survived == 1)
    g <- ggplot(data = d2, aes(x=firstname, y=rate))
    g <- g + geom_bar(stat='identity') 
    g <- g + labs(title="Survival rate by firstname", x="Firstname", y="Survival rate in %") 
    g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    g

![](/sites/default/files/basic_page/titanic-15-1.png)

Here we took firstnames that appear at least 5 times (less than that,
it's probabely not relevant), and plot the survival rate. We can see
that Margarets, Elizabeths and Anna have 100% survival rate, whereas
Arthurs, Johan or Joseph are a lot less lucky. We can add this to our
features.

### Passenger sex

Male or female, does it have a correlation with the survival rate ?

    d <- df %>% select(Survived, Sex)
    g <- ggplot(data = d, aes(x=Sex, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Women/Men Survivors", x="Sex", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-16-1.png)

Ok, looks like female have more chance to survive. This is definitely an
important feature for our model.

### Passenger age

We see a few missing values here. Let's convert this column to numeric
and make sure we exclude the NA's values from this analysis.

    df$Age <- as.numeric(as.character(df$Age))

    d <- df %>% select(Age, Survived) %>% filter(!is.na(Age))
    g <- ggplot(data = d, aes(x=Age, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Age of Survivors", x="Age", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-18-1.png)

The passenger age seems a very important feature. We will have to decide
a method for filling in the missing values. A simple linear regression
model to fill in the missing ages based on other parameters (passenger
class, sex, fare...) would probably work fine.

### Number of Siblings/Spouses Aboard

We can se from the data if a passenger has a sibling on board. Does it
impact its survival rate ?

    d <- df %>% select(Survived, SibSp)
    g <- ggplot(data = d, aes(x=SibSp, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors with siblings", x="Number of siblings", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-19-1.png)

It looks like the more siblings on board, the less likely the passenger
will survive.

### Number of Parents/Children Aboard

    d <- df %>% select(Survived, Parch)
    g <- ggplot(data = d, aes(x=Parch, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors with Parents/Children", x="Number of relation", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-20-1.png)

This doesn't look like a very important feature. Maybe that can be used
to derive features, copling with the sex for example. Here are the
survival rates for mens depending on their relationships:

    d <- df %>% 
        select(Sex, Survived, Parch) %>% 
        filter(Sex == 'male') %>% 
        select(Survived, Parch) %>% 
        group_by(Parch, Survived) %>% 
        summarise(n=n()) %>% 
        mutate(rate=n/sum(n) * 100)
    d

    ## Source: local data frame [9 x 4]
    ## Groups: Parch [6]
    ## 
    ##   Parch Survived     n      rate
    ##   (int)   (fctr) (int)     (dbl)
    ## 1     0        0   404  83.47107
    ## 2     0        1    80  16.52893
    ## 3     1        0    39  67.24138
    ## 4     1        1    19  32.75862
    ## 5     2        0    21  67.74194
    ## 6     2        1    10  32.25806
    ## 7     3        0     1 100.00000
    ## 8     4        0     2 100.00000
    ## 9     5        0     1 100.00000

Males with more than 2 relationships on board die for sure. Males with 0
relationships on board are less likely to survive as males with 1 to 2
relationships on board.

### Family ?

Given that we have the names of the passengers and the configuration of
their families, maybe we can try to recompose some families. If a family
member is not in the training set but they all survived, this might tell
us if the remaining family member(s) survived as well. This will only
work if the data is good enough... let's have a look.

    df[df$SibSp == 5,]

    ##     PassengerId Survived Pclass                               Name    Sex
    ## 60           60        0      3 Goodwin, Master. William Frederick   male
    ## 72           72        0      3         Goodwin, Miss. Lillian Amy female
    ## 387         387        0      3    Goodwin, Master. Sidney Leonard   male
    ## 481         481        0      3     Goodwin, Master. Harold Victor   male
    ## 684         684        0      3        Goodwin, Mr. Charles Edward   male
    ##     Age SibSp Parch  Ticket Fare Cabin Embarked
    ## 60   11     5     2 CA 2144 46.9              S
    ## 72   16     5     2 CA 2144 46.9              S
    ## 387   1     5     2 CA 2144 46.9              S
    ## 481   9     5     2 CA 2144 46.9              S
    ## 684  14     5     2 CA 2144 46.9              S

Ok, looks like we got our family Goodwin there. They didn't survived...
but wait, we got 5 members, and they all have 5 siblings and 2 parents,
that would make 8 Goodwin in total. Are they in the test set ?

    df_test[grepl('Goodwin', df_test$Name),]

    ##     PassengerId Pclass                           Name    Sex Age SibSp
    ## 140        1031      3 Goodwin, Mr. Charles Frederick   male  40     1
    ## 141        1032      3    Goodwin, Miss. Jessie Allis female  10     5
    ##     Parch  Ticket Fare Cabin Embarked
    ## 140     6 CA 2144 46.9              S
    ## 141     2 CA 2144 46.9              S

hmm that's 2 more. And Mr. Charles Frederick is the father, no trace of
the spouse, which is also supposed to be on board.

Logically, she wouldn't have 5 siblings, but 6 children like her
husband. Is she in the train set ?

    df[df$Parch == 6,]

    ##     PassengerId Survived Pclass                                    Name
    ## 679         679        0      3 Goodwin, Mrs. Frederick (Augusta Tyler)
    ##        Sex Age SibSp Parch  Ticket Fare Cabin Embarked
    ## 679 female  43     1     6 CA 2144 46.9              S

Bingo. And we know she also died.

Ok so probably reconstructing the families may help our model to predict
with more accuracy if the passenger survived or not. Let's see is the
whole data is good enough for data.

How many families do we have:

    d <- df %>% mutate('familyname' = as.factor(gsub("^([^,]*),.*", "\\1", Name, perl=TRUE)))
    length(levels(d$familyname))

    ## [1] 667

    d %>% filter(SibSp > 0 | Parch > 0) %>% count

    ## Source: local data frame [1 x 1]
    ## 
    ##       n
    ##   (int)
    ## 1   354

So we can split the 891 passengers into 667 families. 354 of which are
"real" families and not lonesome travellers. We can create new features
using this information.

Chances are, that new families will show up in the test set and that
some families are all in the training set. So we'll have to account for
that and add a neutral feature "unkonwn\_family".

Just to make sure we don't have duplicated names, let's look at the
family names with the more members.

    d %>% group_by(familyname) %>% summarise(no = length(familyname)) %>% filter(no > 4)

    ## Source: local data frame [8 x 2]
    ## 
    ##   familyname    no
    ##       (fctr) (int)
    ## 1  Andersson     9
    ## 2     Carter     6
    ## 3    Goodwin     6
    ## 4    Johnson     6
    ## 5     Panula     6
    ## 6       Rice     5
    ## 7       Sage     7
    ## 8      Skoog     6

Ok, looks like we have 9 Andersson on board, let's look at this family:

    d %>% filter(familyname == 'Andersson') %>% select(Name, Age, Parch, SibSp, Cabin, Embarked) %>% arrange(desc(Age))

    ##                                                        Name Age Parch
    ## 1                               Andersson, Mr. Anders Johan  39     5
    ## 2 Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)  39     5
    ## 3              Andersson, Mr. August Edvard ("Wennerstrom")  27     0
    ## 4                           Andersson, Miss. Erna Alexandra  17     2
    ## 5                         Andersson, Miss. Sigrid Elisabeth  11     2
    ## 6                      Andersson, Miss. Ingeborg Constanzia   9     2
    ## 7                        Andersson, Miss. Ebba Iris Alfrida   6     2
    ## 8                   Andersson, Master. Sigvard Harald Elias   4     2
    ## 9                         Andersson, Miss. Ellis Anna Maria   2     2
    ##   SibSp Cabin Embarked
    ## 1     1              S
    ## 2     1              S
    ## 3     0              S
    ## 4     4              S
    ## 5     4              S
    ## 6     4              S
    ## 7     4              S
    ## 8     4              S
    ## 9     4              S

We have the parents first, they have 5 children on board. Mrs Anders
Johan is traveling alone and does not belong to this family. Then we
have 6 kids. 6? They're only suppose to have 4 siblings, which would
make family of 7 (5 kids and 2 parents).

In the test set we have 2 Andersson's:

    dt <- df_test %>% mutate('familyname' = as.factor(gsub("^([^,]*),.*", "\\1", Name, perl=TRUE)))
    dt %>% filter(familyname == 'Andersson') %>% select(Name, Age, Parch, SibSp, Cabin, Embarked)

    ##                                     Name Age Parch SibSp Cabin Embarked
    ## 1 Andersson, Miss. Ida Augusta Margareta  38     2     4              S
    ## 2            Andersson, Mr. Johan Samuel  26     0     0              S

That's another lonesome traveller (Mr Johan Samuel) and another one with
also 4 siblings/2 parents, and she's as old as the parents we thought we
identified.

We can't separate them, even looking at the class, port or cabin...
Looking at the data information file from Kaggle, the relationships are
actually a bit vague:

    Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
    Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
    Parent:   Mother or Father of Passenger Aboard Titanic
    Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

But a more important issue is to separate the lonesome travellers from
the rest of the families. That's actually easy looking at `SibSp` and
`Parch`. We'll make sure to separate these in the preprocessing phase.
We can build the family name feature by including the supposed number of
family members (by adding SipSp, Parch + 1). Our family features will
therefore have the form `<family_name><number_of_members>`. In the case
of the Andersson, we'll end up with 9 members for a supposedly 7 members
family. That's not ideal, but I don't see how to make it better.

So with the family name we derived the following boolean features:

-   title (on feature by title)
-   bracket in name
-   belong to family (as many features as families)
-   belong to an unknown family
-   lonesome traveller

### Ticket

This feature is a bit more difficult to analyse. The format is not well
defined, and its significance is difficult to establish. Let's look at
some information from the ticket references, by splitting the numbers
and strings.

    d <- df %>% 
        mutate(
            TicketString = toupper(gsub('\\.?\\s?/?\\d?', '', Ticket)),
            TicketNumber = gsub("\\D", "", Ticket)
        )
    unique(d$TicketString)

    ##  [1] "A"         "PC"        "STONO"     ""          "PP"       
    ##  [6] "CA"        "SCPARIS"   "SCA"       "SP"        "SOC"      
    ## [11] "WC"        "SOTONOQ"   "WEP"       "C"         "SOP"      
    ## [16] "FA"        "LINE"      "FCC"       "SWPP"      "SCOW"     
    ## [21] "PPP"       "SC"        "SCAH"      "AS"        "SCAHBASLE"
    ## [26] "SOPP"      "FC"        "SOTONO"    "CASOTON"

That's going to be difficult to use... let's see the survival rates for
each reference that we have.

    g <- ggplot(data = d, aes(x=TicketString, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors by ticket string", x="Ticket string", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    g

![](/sites/default/files/basic_page/titanic-30-1.png)

I'm not sure I'll use this feature in my model.

Maybe check only the survival rate difference between passenger with
letters in their tickets against passenger with only numbers:

    d %>% 
        filter(TicketString == "") %>% 
        group_by(Survived) %>% 
        summarise(n = n()) %>% 
        mutate(survivalrate = n / sum(n))

    ## Source: local data frame [2 x 3]
    ## 
    ##   Survived     n survivalrate
    ##     (fctr) (int)        (dbl)
    ## 1        0   407    0.6157337
    ## 2        1   254    0.3842663

    d %>% 
        filter(TicketString != "") %>% 
        group_by(Survived) %>% 
        summarise(n = n()) %>% 
        mutate(survivalrate = n / sum(n))

    ## Source: local data frame [2 x 3]
    ## 
    ##   Survived     n survivalrate
    ##     (fctr) (int)        (dbl)
    ## 1        0   142    0.6173913
    ## 2        1    88    0.3826087

Ok.. not much difference here between ticket holders with and without
strings in their ticket. Maybe the ticket number is more useful. let's
convert it to a numeric and plot the survival rate:

    d$TicketNumber <- as.numeric(d$TicketNumber)
    g <- ggplot(data = d, aes(x=TicketNumber, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors by ticket number", x="Ticket number", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-32-1.png)

Here we can see that ticket holders with a small ticket numbers are
among the vast majority of the passengers. But this doesn't help. Maybe
we can check the first digit of the ticket, this may be related to the
location on the boat.

    d <- d %>%
        mutate('TicketDigit' = as.factor(gsub('(\\d)\\d*\\$?', '\\1', TicketNumber, perl=TRUE)))
    g <- ggplot(data = d, aes(x=TicketDigit, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors by ticket number", x="Ticket number first digit", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-33-1.png)

It looks like the first digit is actually relevant. The lower this
number, the higher the survival rate.

We'll see if we can use all these information when preprocessing the
data in the next steps. For now, that's all the information I can draw
from the Ticket feature.

### Fare

That's also probabely a proxy for the passenger class, hence for the
social status. First let's convert it to numeric. We have 0.00 values,
which seem a bit strange.

Let's ignore these for this exploration. We can fill them later on if
needed. FOr example, using the mean of the class fare, that would work.

    d <- df %>% filter(Fare != 0)
    d$Fare <- as.numeric(as.character(d$Fare))
    g <- ggplot(data = d, aes(x=Fare, fill=Survived))
    g <- g + geom_bar() 
    g <- g + labs(title="Survivors and fares", x="Ticket fare", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-34-1.png)

Also 3 passengers paid more than 400, and they all survived. Let's look
in more details at the lower fares.

    g <- ggplot(data = d %>% filter(Fare < 100), aes(x=Fare, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors and fares", x="Ticket fare", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-35-1.png)

Clearly, a big proportion of the lower fares died. We cannot ignore this
feature for our model, maybe we can fill in the missing values by using
the median price (the mean is a bit skewed because of those very
expensive passengers).

### Cabin number

This one is a bit complicated because of the amount of missing values.

    (df %>% filter(Cabin == "") %>% count)$n

    ## [1] 687

So.. we can still have a look, maybe at the floor (assuming this is the
meaning of the letter) and the survival rate for each floor.

    d <- df %>%
        filter(Cabin != "") %>%
        mutate(Cabin = gsub("^(\\w).*$", "\\1", Cabin)) %>%
        select(Cabin, Survived)
    g <- ggplot(data = d, aes(x=Cabin, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors by Cabin", x="Cabin floor", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-37-1.png)

    dd <- d %>% 
        group_by(Cabin, Survived) %>% 
        summarise(n = n()) %>% 
        mutate(survivalrate = n / sum(n)) %>%
        select(Survived, Cabin, survivalrate) %>%
        spread(Survived, survivalrate)
    colnames(dd) <-  c('floor', 'died', 'survived')
    dd

    ## Source: local data frame [8 x 3]
    ## 
    ##   floor      died  survived
    ##   (chr)     (dbl)     (dbl)
    ## 1     A 0.5333333 0.4666667
    ## 2     B 0.2553191 0.7446809
    ## 3     C 0.4067797 0.5932203
    ## 4     D 0.2424242 0.7575758
    ## 5     E 0.2500000 0.7500000
    ## 6     F 0.3846154 0.6153846
    ## 7     G 0.5000000 0.5000000
    ## 8     T 1.0000000        NA

If the letter indicates the floor or deck, maybe the number indicates
the position of the cabin.

    d <- df %>%
        filter(Cabin != "") %>%
        filter(grepl("\\d", Cabin)) %>%
        mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
        select(Cabin, Survived) %>%
        arrange(Cabin)
    g <- ggplot(data = d, aes(x=Cabin, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors by Cabin", x="Cabin position", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    g

![](/sites/default/files/basic_page/titanic-39-1.png)

We can maybe see a pattern here, with chunks of full survival and chunks
of full not-survival. It's not a linear progression, so we can group the
position into bins to make it easier to use.

    d <- df %>%
        filter(Cabin != "") %>%
        filter(grepl("\\d", Cabin)) %>%
        mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
        mutate(Position = cut(Cabin, 10)) %>%
        select(Position, Survived) %>%
        arrange(Position)
    g <- ggplot(data = d, aes(x=Position, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors by Cabin", x="Cabin position", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    g

![](/sites/default/files/basic_page/titanic-40-1.png)

Ok that can be used in the model. We would need to know the maximal
cabin number from the test set as well to make sure we can correctly
classify the cabins in the bins.

    df %>% 
        filter(Cabin != "") %>%
        filter(grepl("\\d", Cabin)) %>%
        mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
        top_n(1, Cabin) %>% select(Cabin)

    ##   Cabin
    ## 1   148

    df_test %>% 
        filter(Cabin != "") %>%
        filter(grepl("\\d", Cabin)) %>%
        mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
        top_n(1, Cabin) %>% select(Cabin)

    ##   Cabin
    ## 1   132

Maybe just the fact to have a cabin number means you have more chance to
survive ?

    dd <- df %>% 
        mutate(hascabin = Cabin != "") %>% 
        group_by(hascabin, Survived) %>% 
        summarise(n = n()) %>% 
        mutate(survivalrate = n / sum(n)) %>% 
        select(hascabin, Survived, survivalrate) %>% 
        spread(Survived, survivalrate)
    colnames(dd) <-  c('has cabin', 'died', 'survived')
    dd

    ## Source: local data frame [2 x 3]
    ## 
    ##   has cabin      died  survived
    ##       (lgl)     (dbl)     (dbl)
    ## 1     FALSE 0.7001456 0.2998544
    ## 2      TRUE 0.3333333 0.6666667

ok, we don't have any cabin number &gt; 148, so we can use the train set
range to create the bins.

Well looks like having a cabin number gives you a very high chance of
survival (66%) compard to not having one (30%). So we can probabely
change this feature to a simple boolean value.

Also the number of cabin may matter:

    d <- df %>%
        mutate('cabinCount' = as.factor(sapply(gregexpr("[[:alpha:]]+", Cabin), function(x) sum(x > 0)))) %>%
        select(cabinCount, Survived)
    g <- ggplot(data = d, aes(x=cabinCount, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors by Cabin count", x="Cabin count", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-43-1.png)

Indeed, the more cabins, the more chance of survival.

### Port of Embarkation

This feature is a factor with 3 levels:

    levels(df$Embarked)

    ## [1] ""  "C" "Q" "S"

With C = Cherbourg, Q = Queenstowna dn S = Southampton. 2 Passenger have
no information regarding their port of embarkation, they both survived.
I don't think that's relevant.

Let's look at the survival rate per port:

    d <- df %>% filter(Embarked != "") %>% select(Survived, Embarked)
    g <- ggplot(data = d, aes(x=Embarked, fill=Survived))
    g <- g + geom_bar(binwidth=1) 
    g <- g + labs(title="Survivors by Port", x="Port of Embarkation", y="Number of passengers") 
    g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
    g

![](/sites/default/files/basic_page/titanic-45-1.png)

It's not really evident if the port of embarkation has an impact on the
passenger's faith. Looks like better odds for passenger coming from
Cherbourg.

    dd <- df %>% 
        filter(Embarked != "") %>% 
        select(Survived, Embarked) %>% 
        group_by(Embarked, Survived) %>% 
        summarise(n = n()) %>% 
        mutate(survivalrate = n / sum(n)) %>% 
        select(Embarked, Survived, survivalrate) %>% 
        spread(Survived, survivalrate)
    colnames(dd) <- c("Port or embarkation", "died", "survived")
    dd

    ## Source: local data frame [3 x 3]
    ## 
    ##   Port or embarkation      died  survived
    ##                (fctr)     (dbl)     (dbl)
    ## 1                   C 0.4464286 0.5535714
    ## 2                   Q 0.6103896 0.3896104
    ## 3                   S 0.6630435 0.3369565

So we can see that passengers embarked in Cherbourg have 55% chance of
surviving, that's pretty good compared to the 2 other embarcation ports.
