options(java.parameters = "-Xmx30g")
library(data.table)
library(h2o)
library(DataExplorer)
library(tm)
library(dplyr)

# Source data came from https://www.kaggle.com/codename007/funding-successful-projects#train.csv


raw_data <- read.csv("~/Kickstarter_Success/train.csv", stringsAsFactors = FALSE)



# Convert unix date/timestamp to date format
raw_data$deadline <- as.Date(as.POSIXct(as.numeric(as.character(raw_data$deadline)),origin="1970-01-01",tz="GMT"), format='%Y-%m-%d %H:%M:$S')
raw_data$state_changed_at <- as.Date(as.POSIXct(as.numeric(as.character(raw_data$state_changed_at)),origin="1970-01-01",tz="GMT"), format='%Y-%m-%d %H:%M:$S')
raw_data$created_at <- as.Date(as.POSIXct(as.numeric(as.character(raw_data$created_at)),origin="1970-01-01",tz="GMT"), format='%Y-%m-%d %H:%M:$S')
raw_data$launched_at <- as.Date(as.POSIXct(as.numeric(as.character(raw_data$launched_at)),origin="1970-01-01",tz="GMT"), format='%Y-%m-%d %H:%M:$S')

# Convert all currency to USD
conversionDf <- data.frame(currency = c('AUD','CAD','DKK','EUR','GBP','NOK','NZD','SEK','USD'),
                           conversion_rate = c(0.71, 0.75, 0.15, 1.13, 1.33, 0.12, 0.68, 0.11, 1))
conversionDf$currency <- as.character(conversionDf$currency)
conversionDf <- as.data.table(conversionDf)

raw_data <- as.data.table(raw_data)
kickstartDf <- merge(raw_data, conversionDf, by='currency', all.x=TRUE)
kickstartDf$goal_normalized <- round(kickstartDf$goal*kickstartDf$conversion_rate)


# Create goal dollars per day, as defined as the time between 'deadline' and 'launched_at'
kickstartDf$num_days <- as.integer(kickstartDf$deadline - kickstartDf$launched_at)
kickstartDf$dollars_per_day <- round(kickstartDf$goal_normalized/kickstartDf$num_days)

# Make a vector source: coffee_source
keyword_vector <- gsub("-", " ", kickstartDf$keywords)

keyword_vector <- keyword_vector[1:(length(keyword_vector)*0.1)]

keyword_source <- VectorSource(keyword_vector)

keyword_corpus <- VCorpus(keyword_source)

keyword_tdm <- TermDocumentMatrix(keyword_corpus)

## coffee_tdm is still loaded in your workspace

# Create a matrix: coffee_m
keyword_m <- as.matrix(keyword_tdm)

# Calculate the rowSums: term_frequency
term_frequency <- rowSums(keyword_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency, decreasing=TRUE)

# View the top 10 most common words
term_frequency[1:25]

# Continue to remove non-descript patterns until only informative keywords remain
keyword_vector <- gsub("and|the|project|short|debut|feature|love|story|for|new|help|first|with|you|make|from|about|release|full|length|life|2011|get|web|our|man|one|making|part|out|fund","", keyword_vector)
keyword_source <- VectorSource(keyword_vector)
keyword_corpus <- VCorpus(keyword_source)
keyword_tdm <- TermDocumentMatrix(keyword_corpus)
keyword_m <- as.matrix(keyword_tdm)
term_frequency <- rowSums(keyword_m)
term_frequency <- sort(term_frequency, decreasing=TRUE)
term_frequency[1:20]

# Plot a barchart of the 10 most common words
barplot(term_frequency[1:20], col="tan", las=2)


# Create 'subject' variable which indicates the primary key word used in the description of the kickstarter fund
kickstartDf$subject <- ifelse(grepl("album|Album|music|Music", kickstartDf$keywords), "MUSIC",
                              ifelse(grepl("film|Film|Movie|movie", kickstartDf$keywords), "MOVIE",
                                     ifelse(grepl("series|Series", kickstartDf$keywords), "SERIES",
                                            ifelse(grepl("art|Art", kickstartDf$keywords), "ART",
                                                   ifelse(grepl("book|Book|novel|Novel", kickstartDf$keywords), "BOOK",
                                                          ifelse(grepl("documentary|Documentary", kickstartDf$keywords), "DOCUMENTARY",
                                                                 ifelse(grepl("video|Video", kickstartDf$keywords), "VIDEO",
                                                                        ifelse(grepl("game|Game", kickstartDf$keywords), "GAME",
                                                                               ifelse(grepl("comedy|Comedy", kickstartDf$keywords), "COMEDY", NA)))))))))

# subset data to only variables we would be interested in potentially using in our model
trainVars <- c('project_id','goal_normalized','disable_communication','country','backers_count','num_days','dollars_per_day','subject','final_status')
kickstartDf <- as.data.frame(kickstartDf)
kickstartSubset <- kickstartDf[, trainVars]


# 'backers_count' is the only variable that appears to have a moderate correlation to our 'final_status' predictor variable
plot_correlation(kickstartSubset)


# Data set is unbalanced, so we will need to perform upsampling during our grid search
table(kickstartSubset$final_status)


# Convert non-numerical data points to factors
factorVars <- c('disable_communication','country','subject','final_status')
kickstartSubset[factorVars] <- lapply(kickstartSubset[factorVars], factor)


# Determine the frequency of missing values per field
plot_missing(kickstartSubset)


# Split into training/validation and testing sets
trainValidation <- kickstartSubset[1:(nrow(kickstartSubset)*0.85),]
testDf <- kickstartSubset[(nrow(kickstartSubset)*0.85):nrow(kickstartSubset),]


# Split data into training and validation data sets
h2o.init()

trainh2o <- as.h2o(trainValidation)
testh2o <- as.h2o(testDf)
splits <- h2o.splitFrame(data= trainh2o, ratios = 0.75, seed = 1234)
train <- splits[[1]]
validation <- splits[[2]]

gbm_params <- list(learn_rate = 0.01,
                   max_depth = c(8,12),
                   ntrees = c(250, 400),
                   min_rows = c(5,10),
                   balance_classes = TRUE)

x <- c('goal_normalized','disable_communication','country','backers_count','num_days','dollars_per_day','subject')
y <- 'final_status'

gbm_grid <- h2o.grid(x = x, y = y,
                     training_frame = train,
                     validation_frame = validation,
                     hyper_params = gbm_params,
                     algorithm = "gbm",
                     grid_id = "kickstarter_grid",
                     seed= 1234)

sorted_grid <- h2o.getGrid(grid_id = gbm_grid@grid_id, sort_by="AUC", decreasing=TRUE)
print(sorted_grid)

# Grab the top GBM model, chosen by validation AUC
best_gbm1 <- h2o.getModel(sorted_grid@model_ids[[1]])

# Measure GBM model performance on test set
best_gbm_perf1 <- h2o.performance(model = best_gbm1,
                                  newdata = testh2o)
h2o.auc(best_gbm_perf1)
# Produces an AUC of 0.9680745

# Predict on test set and measure accuracy using confusion matrix
h2o.confusionMatrix(best_gbm_perf1)
# Produces a Recall Score of 88%
# and a Specificity score of 92%
# and a Precision rating of 77%
# and an F1 Score of 82.1%

# Plot variable importance
h2o.varimp_plot(best_gbm1, num_of_features = 10)






# Create logistic regression using same training and validation data sets
glm_model <- h2o.glm(x = x, 
                     y = y, 
                     training_frame = train,
                     validation_frame = validation, 
                     seed = 1234,        # Make sure same seed is used as previous model
                     family = "binomial",   
                     lambda_search = TRUE,  # Optimum regularisation lambda
                     balance_classes = TRUE
)

# Validation data set accuracy (AUC)
glm_model@model$validation_metrics@metrics$AUC
# Logistic model generates an AUC of 0.8077306 on the validation set

# Measure model GLM performance on test set
glm_performance <- h2o.performance(model = glm_model,
                                   newdata = testh2o)

h2o.auc(glm_performance)
# Produces an AUC of 0.8803654 on the test set

# Predict on test set and measure accuracy using confusion matrix
h2o.confusionMatrix(glm_performance)
# Produces a Recall Score of 76%
# and a Specificity score of 84%
# and a Precision rating of 60%
# and an F1 Score of 67.1%

#compute variable importance and performance
h2o.varimp_plot(glm_model, num_of_features = 10)


# Ultimately we found that our GBM model performs noticeably better on our test set when 
# compared to our Logistic Regression model as measured by; Recall (Sensitivity), Precision,
# Specificity, F1 Score, and AUC


h2o.shutdown(prompt = FALSE)