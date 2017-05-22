## Churn prediction of Uber customer
### ### Aim of the project:  
Develop a model to predict the churn of Uber customer based on their behavioral data.

### Data source:  
Uber customer behavioral data.

### Data analysis pipeline:  
  1. Data cleaning  
     1.1 Remove invalid and duplicated cases  
     1.2 Deal with missing data  
          a) Fill missing categorical entries with new value - 'Missing value'  
          b) Imputation of missing customer rating with the average rating of subgroup of customer in the training dataset

  2. Feature engineering  
     2.1 Generation of feature 'weekend ride', 'weekday ride', 'average spending per ride'   

  3. Model development  
     3.1 Linear regression model as the baseline model  
     3.2 Random forest regression model  
     3.3 Gradient boosting regression model  

### Model evaluation:   
The final gradient boosting model has accuracy score of 0.79, precision score of 0.81, and recall score of 0.86.  
The ROC curve.
![alt text](https://github.com/haowei772/Case_study_02/blob/master/figures/ROC_curve.png)

The model revealed that features with high impact on customer churn are the rating of the customer and driver, distance of the ride, percentage of the surge ride, and promotion period.
Feature importance  
![alt text](https://github.com/haowei772/Case_study_02/blob/master/figures/Feature_importance.png)
