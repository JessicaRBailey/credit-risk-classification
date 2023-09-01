# credit-risk-classification
Challenge 20

In this project, we have a set of data that contains historical loan details from a peer-to-peer lending services company.  Our goal is to help the company identify the creditworthiness of borrowers.  If the  model is good, the company can use this to make better decisions about lending to borrowers in the future.  We use supervised machine learning, specifically logistical regression, to train and evaluate a model based on loan risk. 

This dataset, lending_data.csv, contains information about each borrower the company has already lent to, and importantly, whether or not they defaulted on their loan.  We will be using whether they defaulted as our y, or dependent variable.  The other information collected on each borrower includes data such as:
* loan size	
* interest rate
* borrower's income
* debt to income ratio
* number of accounts
* whether there are any derogatory marks
* total debt
These features will comprise our X dataset and will be used to predict y.  

When we do a cursory analysis of the dataset, we see that significantly more people repaid their loan (75036) than did not (2500).  This has the potential to weaken the models ability to predict on the smaller dataset, which is the one we are most concerned about, so we will watch for this in our model.  

To prepare our dataset for logistcal regression modeling, we split it into a training set and a testing set. Then we set up the model using (solver='lbfgs', max_iter=200, random_state=1), and used the training sets to fit, or train the model and finally predict.  

In a second analysis, we also used RandomOverSampler to create equally sized X and y data sets, to attempt to improve performance on our default predictions.  

We used balanced accuracy score, confusion matrix, and classification report to analyze our model.  


## Results

* Machine Learning Model 1:
  * Balanced Accuracy: 0.952
  * Precision
    * Healthy Loans: 1.00
    * High-risk Loans: 0.85
  * Recall
    * Healthy Loans: 0.99
    * High-risk Loans: 0.91

* Machine Learning Model 2 (resampled):
  * Balanced Accuracy: 0.994
  * Precision
    * Healthy Loans: 1.00
    * High-risk Loans: 0.84
  * Recall
    * Healthy Loans: 0.99
    * High-risk Loans: 0.99

## Summary

Both models performed well with all scores above 80% and many near 100%, however, the model using the resampled dataset performed better, achieving a balanced accuracy score of 0.99 as compared to 0.95 with model 1.  The improvement came in the detection of high-risk loans.  The rebalanced dataset allowed the model to reduce the false negatives from 56/563 (9.95%) to 4/615 (0.65%).  Because the lending company is more concerned about avoiding lending to those who are likely to default, it is recommended that they continue to use the logistical regresssion model with resampled data to better avoid approving high-risk loans.  


