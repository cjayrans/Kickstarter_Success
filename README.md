# Kickstarter Success

Kickstarter is a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing creative project to life. Till now, more than $3 billion dollars have been contributed by the members in fueling creative projects. The projects can be literally anything – a device, a game, an app, a film etc. Kickstarter had released it's public repository to help answer the question, will a project meet its funding goal? 

Further information can be found on [Kaggle](https://www.kaggle.com/codename007/funding-successful-projects#train.csv). In order to predict the successful funding of a Kickstarter project we must perform an extensive amount of data cleansing, creation of engineered vairables, as well as the use of the Bag-of-words model to prepare the data for modeling. Once compiled, we test out the performance of a Logistic Regression and a Gradient Boosted Machine algorithm to perform our classifications. We also perform upsampling as proportionally fewer projects ever meet their funding goal. 

When doing so we found that our GBM model performs noticeably better than our Logistic Regression. Listed below are the accuracy measurements when testing each model on our test set.
1. GBM Model Performance
   - Recall: 88%
   - Specificity: 92%
   - Precision: 77%
   - F1 Score: 82%
2. Logistic Regression Model Performance
   - Recall: 76%
   - Specificity: 84%
   - Precision: 60%
   - F1 Score: 67%
