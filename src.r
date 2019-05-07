# Author: Sabirah Shuaybi 2019

library(readr)
library(dplyr)
library(ggplot2)
library(purrr)
library(caret)
library(rpart)

# Reading in the 3 Data Sets (to be merged)

data <- read_csv("~/DataScience/2018VAERSDATA.csv")
symptoms <- read_csv("~/DataScience/2018VAERSSYMPTOMS.csv")
vax <- read_csv("~/DataScience/2018VAERSVAX.csv")

# Merging Data Set

vaers <- merge(data, symptoms, by = "VAERS_ID")
vaers <- merge(vaers, vax, by = "VAERS_ID")

# Now vaers is the fully merged data set

# Subsetting Data (only keeping useful predictors in)
vaers <- vaers %>%
  select(AGE_YRS, SEX, DIED, ER_VISIT, HOSPDAYS, RECOVD, V_ADMINBY, VAX_MANU, VAX_ROUTE, VAX_SITE, VAX_TYPE)


### Data Clean Up


# Imputation of Missing Age Data (using median)
impute_missing_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
}
vaers <- vaers %>% mutate_at("AGE_YRS", impute_missing_median)

#check if all the missing values has been filled in
sum(is.na(vaers$AGE_YRS))

vaers$DIED[is.na(vaers$DIED)] <- "N"
vaers$ER_VISIT[is.na(vaers$ER_VISIT)] <- "N"
vaers$HOSPDAYS[is.na(vaers$HOSPDAYS)] <- 0
vaers$RECOVD[is.na(vaers$RECOVD)] <- "U"
vaers$VAX_ROUTE[is.na(vaers$VAX_ROUTE)] <- "UN"
vaers$VAX_SITE[is.na(vaers$VAX_SITE)] <- "UN"

vaers <- vaers %>%
     mutate(
        VAX_TYPE = factor(VAX_TYPE)
     )

str(vaers)



#Keeping: FLU3 MMR TDAP VARCEL HPV9 HEP

vaers <- vaers %>% filter(VAX_TYPE == c("FLU3", "MMR", "TDAP", "VARCEL", "HPV9", "HEP"))

distinct(vaers, VAX_TYPE)
str(vaers)


### Data Exploration and Visualizations


vaers %>% group_by(VAX_TYPE) %>% tally()
vaers %>% group_by(VAX_MANU) %>% tally()
vaers %>% group_by(HOSPDAYS) %>% tally()


ggplot(vaers, mapping = aes(x = VAX_TYPE, fill = VAX_MANU)) +
  geom_bar() +
  coord_flip()

ggplot(vaers, mapping = aes(x = VAX_TYPE, fill = SEX)) +
  geom_bar() +
  coord_flip()

ggplot(vaers, mapping = aes(x = VAX_MANU)) +
  geom_bar() +
  coord_flip()

ggplot(vaers, mapping = aes(x = VAX_TYPE, y = AGE_YRS)) +
  geom_boxplot() +
  coord_flip()

ggplot(vaers[which(vaers$HOSPDAYS>0),], mapping = aes(x = AGE_YRS, y = HOSPDAYS, color = RECOVD)) +
  geom_point()


### Setting up Train/Test Split and Cross-Validation Folds

set.seed(723)

# Train/test split
tt_inds <- caret::createDataPartition(vaers$VAX_TYPE, p = 0.7)
train_set <- vaers %>% slice(tt_inds[[1]])
test_set <- vaers %>% slice(-tt_inds[[1]])


### Fit and test set classification error rate via multinomial logistic regression


multilogistic_fit <- train(
  VAX_TYPE ~ .,
  data = train_set,
  trace = FALSE,
  method = "multinom",
  trControl = trainControl(
    method = "cv",
    number = 10,
    returnResamp = "all",
    savePredictions = TRUE,
),
tuneGrid = data.frame(decay = seq(from = 0, to = 0.2, length = 30))
)

# Pick tuning parameter values yielding highest cross-validated accuracy
multilogistic_fit$results %>% filter(Accuracy == max(Accuracy))


### Test set performance: looking for low test set error rate, high test set accuracy


mean(test_set$VAX_TYPE != predict(multilogistic_fit, test_set))
mean(test_set$VAX_TYPE == predict(multilogistic_fit, test_set))


### Fit and test set classification error rate via multi-class gradient boosting

xgb_fit <- train(
  VAX_TYPE ~ .,
  data = train_set,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 10,
    returnResamp = "all",
    savePredictions = TRUE
),
  tuneGrid = expand.grid(
    nrounds = c(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    eta = c(0.5, 0.6, 0.7), # learning rate; 0.3 is the default
    gamma = 0, # minimum loss reduction to make a split; 0 is the default
    max_depth = 1:5, # how deep are our trees?
    subsample = c(0.5, 0.9, 1), # proportion of observations to use in growing each tree
    colsample_bytree = 1, # proportion of explanatory variables used in each tree
    min_child_weight = 1 # think of this as how many observations must be in each leaf node
  )
)

xgb_fit$results %>% filter(Accuracy == max(Accuracy))



# Evaluating Test Set Error Rate and Accuracy Rate

mean(test_set$VAX_TYPE != predict(xgb_fit, test_set))
mean(test_set$VAX_TYPE == predict(xgb_fit, test_set))
