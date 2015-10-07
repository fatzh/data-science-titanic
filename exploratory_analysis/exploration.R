# this is the external script for the exploration Rmarkdown document

# load packages

## ---- loadPackages --------------
library('ggplot2')
library('dplyr')
library('tidyr')

## ----  loadData ------------
df <- read.csv2('../data/train.csv', sep = ',')
df_test <- read.csv2('../data/test.csv', sep=',')

## ---- plotSurvived
df$Survived <- as.factor(df$Survived)
survived <- count(df[df$Survived == 1,])$n
died <- count(df[df$Survived == 0,])$n
survival_rate <- round(survived / (survived + died) *100, 2)
d <- data.frame(
    x = factor(c('Survived', 'Died')),
    y = c(count(df[df$Survived == 1,])$n, count(df[df$Survived == 0,])$n)
)
d
g <- ggplot(d, aes(x=x, y=y))
g <- g + geom_bar(stat='identity', aes(fill=x))
g <- g + labs(title="Titanic survivors", x='', y='')

## ---- plotClass
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

## ---- plotClassSurvival
d <- df %>% select(Survived, Pclass)
g <- ggplot(data = d, aes(x=Pclass, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors by passenger class", x="Passenger class", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- convertNames
df$Name <- as.character(df$Name)

## ---- plotNamesWithBrackets
d <- df %>% mutate(AdditionalName = grepl('\\(', Name)) %>% select(Survived, AdditionalName) 
g <- ggplot(data = d, aes(x=AdditionalName, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors with brackets in their names", x="Brackets in name", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotNamesWithQuotes
d <- df %>% mutate(QuotedName = grepl('\\"', Name)) %>% select(Survived, QuotedName) 
g <- ggplot(data = d, aes(x=QuotedName, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors with quotes in their names", x="Quotes in name", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- extractTitles
titles <- as.factor(gsub(".*,\\s?\\w*?\\s?(\\w+)\\..*", "\\1", df$Name, perl=TRUE))
test_titles <- as.factor(gsub(".*,\\s?\\w*?\\s?(\\w+)\\..*", "\\1", df_test$Name, perl=TRUE))

## ---- groupTitles
titles[titles %in% c('Rev', 'Dr', 'Jonkheer', 'Major', 'Master', 'Capt', 'Col', 'Don')] = 'Sir'
titles[titles %in% c('Countess', 'Dona')] = 'Lady'
titles[titles %in% c('Mlle', 'Ms')] = 'Miss'
titles[titles == 'Mme'] = 'Mrs'
titles = droplevels(titles)

## ---- plotTitles
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

## ---- commonFirstnames
firstname <- as.factor(gsub(".*\\. o?f? ?\\(?(\\w+).*", "\\1", df$Name, perl=TRUE))
d <- df %>% mutate(firstname = firstname)
common_firstnames <- d %>% 
    group_by(firstname) %>% 
    summarise(count=n()) %>% 
    arrange(desc(count)) %>% 
    filter(count > 5) %>%
    select(firstname)

## ---- plotFirstnamesSurival
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

## ---- plotSex
d <- df %>% select(Survived, Sex)
g <- ggplot(data = d, aes(x=Sex, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Women/Men Survivors", x="Sex", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- convertAge
df$Age <- as.numeric(as.character(df$Age))

## ---- plotAge
d <- df %>% select(Age, Survived) %>% filter(!is.na(Age))
g <- ggplot(data = d, aes(x=Age, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Age of Survivors", x="Age", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotSiblings
d <- df %>% select(Survived, SibSp)
g <- ggplot(data = d, aes(x=SibSp, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors with siblings", x="Number of siblings", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotParents
d <- df %>% select(Survived, Parch)
g <- ggplot(data = d, aes(x=Parch, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors with Parents/Children", x="Number of relation", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- parentsSurvival
d <- df %>% 
    select(Sex, Survived, Parch) %>% 
    filter(Sex == 'male') %>% 
    select(Survived, Parch) %>% 
    group_by(Parch, Survived) %>% 
    summarise(n=n()) %>% 
    mutate(rate=n/sum(n) * 100)

## ---- splitFamilyNames
d <- df %>% mutate('familyname' = as.factor(gsub("^([^,]*),.*", "\\1", Name, perl=TRUE)))



## ---- splitTickets
d <- df %>% 
    mutate(
        TicketString = toupper(gsub('\\.?\\s?/?\\d?', '', Ticket)),
        TicketNumber = gsub("\\D", "", Ticket)
    )

## ---- plotTicketStringSurvival
g <- ggplot(data = d, aes(x=TicketString, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors by ticket string", x="Ticket string", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))
g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))

## ---- splitTicketsSurvival
d %>% 
    filter(TicketString == "") %>% 
    group_by(Survived) %>% 
    summarise(n = n()) %>% 
    mutate(survivalrate = n / sum(n))

d %>% 
    filter(TicketString != "") %>% 
    group_by(Survived) %>% 
    summarise(n = n()) %>% 
    mutate(survivalrate = n / sum(n))



## ---- plotTicketNumbers
d$TicketNumber <- as.numeric(d$TicketNumber)
g <- ggplot(data = d, aes(x=TicketNumber, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors by ticket number", x="Ticket number", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotFirstTicketDigit
d <- d %>%
    mutate('TicketDigit' = as.factor(gsub('(\\d)\\d*\\$?', '\\1', TicketNumber, perl=TRUE)))
g <- ggplot(data = d, aes(x=TicketDigit, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors by ticket number", x="Ticket number first digit", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotFares
d <- df %>% filter(Fare != 0)
d$Fare <- as.numeric(as.character(d$Fare))
g <- ggplot(data = d, aes(x=Fare, fill=Survived))
g <- g + geom_bar() 
g <- g + labs(title="Survivors and fares", x="Ticket fare", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotLowestFares
g <- ggplot(data = d %>% filter(Fare < 100), aes(x=Fare, fill=Survived))
g <- g + geom_bar(binwidth=1) 
g <- g + labs(title="Survivors and fares", x="Ticket fare", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- cabinMissing
(df %>% filter(Cabin == "") %>% count)$n

## ---- plotCabinFloor
d <- df %>%
    filter(Cabin != "") %>%
    mutate(Cabin = gsub("^(\\w).*$", "\\1", Cabin)) %>%
    select(Cabin, Survived)
g <- ggplot(data = d, aes(x=Cabin, fill=Survived))
g <- g + geom_bar(binwidth=1) 
g <- g + labs(title="Survivors by Cabin", x="Cabin floor", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- cabinFloorSurvival
dd <- d %>% 
    group_by(Cabin, Survived) %>% 
    summarise(n = n()) %>% 
    mutate(survivalrate = n / sum(n)) %>%
    select(Survived, Cabin, survivalrate) %>%
    spread(Survived, survivalrate)
colnames(dd) <-  c('floor', 'died', 'survived')

## ---- cabinSurvival
dd <- df %>% 
    mutate(hascabin = Cabin != "") %>% 
    group_by(hascabin, Survived) %>% 
    summarise(n = n()) %>% 
    mutate(survivalrate = n / sum(n)) %>% 
    select(hascabin, Survived, survivalrate) %>% 
    spread(Survived, survivalrate)
colnames(dd) <-  c('has cabin', 'died', 'survived')

## ---- cabinCount
d <- df %>%
    mutate('cabinCount' = as.factor(sapply(gregexpr("[[:alpha:]]+", Cabin), function(x) sum(x > 0)))) %>%
    select(cabinCount, Survived)
g <- ggplot(data = d, aes(x=cabinCount, fill=Survived))
g <- g + geom_bar(binwidth=1) 
g <- g + labs(title="Survivors by Cabin count", x="Cabin count", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- plotCabinPositionSurvival
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

## ---- plotCabinPositionBinsSurvival
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

## ---- maxCabinPosition
df %>% 
    filter(Cabin != "") %>%
    filter(grepl("\\d", Cabin)) %>%
    mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
    top_n(1, Cabin) %>% select(Cabin)
df_test %>% 
    filter(Cabin != "") %>%
    filter(grepl("\\d", Cabin)) %>%
    mutate(Cabin = as.numeric(gsub("\\D+(\\d+) ?.*", "\\1", Cabin, perl=TRUE))) %>%
    top_n(1, Cabin) %>% select(Cabin)

## ---- plotPort
d <- df %>% filter(Embarked != "") %>% select(Survived, Embarked)
g <- ggplot(data = d, aes(x=Embarked, fill=Survived))
g <- g + geom_bar(binwidth=1) 
g <- g + labs(title="Survivors by Port", x="Port of Embarkation", y="Number of passengers") 
g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

## ---- portSurvival
dd <- df %>% 
    filter(Embarked != "") %>% 
    select(Survived, Embarked) %>% 
    group_by(Embarked, Survived) %>% 
    summarise(n = n()) %>% 
    mutate(survivalrate = n / sum(n)) %>% 
    select(Embarked, Survived, survivalrate) %>% 
    spread(Survived, survivalrate)
colnames(dd) <- c("Port or embarkation", "died", "survived")