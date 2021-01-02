# DSDJ Salary Prediction Project
Problem Definition: Data on numerous jobs is provided in a csv file. The data contains information such as job title, industry, education, degree, location, years of experience and salary. The task is to explore the data set and use it to develop a model that predicts salaries given different job features. The model is required to have mean squared error(MSE) value less than 360.   

The task has been divided in to 5 steps(each completed in a separate file):

1. DataPrep_salary_pred.ipynb : Data is loaded and cleaned here for EDA
2. EDA_salary_pred.ipynb : Cleaned data is explored to identify patterns and corellations between features and target
3. Prepoc_salary_pred.ipynb : Using insights from EDA, the data is futher processed using ordinal encoding in preparation for modelling
4. Models_salary_pred.ipynb : Linear regression model was explored as a base model. A decision tree model and three ensemble models (random forest, gradient boosting and XGBoost) were further explored. The XGBoost model performed marginally better than the other ensemble methods and was selected as the final model. The models were tuned in a separte file using a grid search to select optimum parameters
5. Deploy_salary_pred.ipynb: In this notebook, loading, cleaning, preprocessing the train data set and training of the selected model on the entire train data set is combined into a single function which outputs an object of the model that is then used to predict the salaries on the test data set. The result is then saved as csv file as requested.

The helper.py file in the module folder contains the functions created and called on at different steps

Areas to Improve on:
1. During EDA over 15000 rows were found with duplicate features but different salaries. This needs further analysis
2. Engineering new features to Improve on MSE
3. Use objected oriented programming to make code more professional




