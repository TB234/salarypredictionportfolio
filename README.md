Salary Prediction Project

## Introduction
Data on job postings are provided in a csv file. Each job posting has a unique identifier(jobId) and information provided for each job include jobtype, degree, major, industry, years of experience and distance from metropolis. Another csv file contains the salary of each job posting. The task is to examine this data set and use it to build a model that predicts the salary on a different job postings data. The model is required to have a mean squared error (MSE) value less than 360 

The task has been divided in to 5 steps(each completed in a separate file):

## 1. Data Cleaning and Processing (file: 1_DataPrep_salary_pred.ipynb) 
The  two csv file (one for features- job information, and and the other for the target- salary) were converted into pandas dataframe objects. Basic summary stats of each data set was carried out which showed  that both data set contained 1000000 rows each which with a unique identifier. The summary stat also showed that the features and target have the right data types. For example years of experience, miles from metropolis and salary  all have intergers values while the values for other features are of object data type. Additionally, the summary stat showed that salary dataframe contained zero value(s). This seems unlikely in reality unless there are volunteer jobs in data set, but no information was given as regards this. The first five rows of each dataframe are shown below

![features_data](./images/features_data.png)



For the features with the object data types, the values were checked for case uniformity as python interprets for example Degree, DEGREE and deGree as different values. The cases were found to be uniform(all uppercases) across all feature values. Next the both dataframes were checked for missing values (including empty strings counting as missing values) and none were found. Both data sets were then checked for duplicates. As expected, there were no duplicates found since each contains a unique job indentifiers in the jobid column. On the features data however, when the jobid column was removed, 8001 duplicates were found (excluding the 'original')

The features and the target dataframes were then merged on the _jobId_ column(with the unique job identifiers). Again we check for duplicates on the merge data after dropping _jobId_ column and now found only 186 duplicates(rows with exact same enteries as another). The fact that we found 8001 duplicates on the features data and only 186 duplicates on the merged data means that there are many cases of jobs with exact same features but with different salaries. This is explored further in EDA.
 The merged data was checked for salary values less that or equal to zero. Five of these were found and the correspoding rows were removed as they provide no useful information and might affect the accuracy of our model. The data was then checked for duplicates. 186 duplicates were still found and these were removed.


![duplicate_Data](./images/features_data.png)
 
## 2. Exploratory Data Analysis (file: EDA_salary_pred.ipynb) 
Here the cleaned data from the previous step is explored to identify patterns and corellations between features and target. As mentioned earlier 8001 duplicate rows were found in th features data and only 186 duplicates found in the merged data, which means that there are many cases of same feature value combinations with different salaries. The table below gives some instances(over 7000 in total) and as can be seen the differences in the salary values are quite significant in some cases. Although this was not carried out but identified as a future work, one could use K Nearest Neighours to select the most accurate of the duplicates. For now what was done was to leave the data as-is and see if it results in a significant different in the performance of our model with the duplicates removed.
Next the distribution of the values in each feature was graphed. This will let us know if our data is balanced and we can be confident of the insights we gather during EDA.We see that the occurence of values in each feature is balanced except for 'NONE' in the majors feature which has about five times more level of occurence. Examining further we deduce that the reason for this is that each degree type other than 'HIGH_SCHOOL' and 'NONE' degree types are distributed uniformly across all major types. 'HIGH_SCHOOL' and 'NONE' degree types exclusively associated with 'NONE' degree.
Next we graph different feature combinations with their averaged salaries. For example we group the data by industry and jobtype and with the graph(shown below) we are able to see the average salaries for each jobtype in a particular industry. From the graph we were able to deduce general trends for each feature:
jobType - salaries increase in the following order: JANITOR, JUNIOR, SENIOR, MANAGER, VICE_PRESIDENT, CFO, CTO, CEO
Degree - salariies increase in the following order: NONE, HIGH_SCHOOL, BACHELORS, MASTERS, DOCTORAL
industry - salaries increase in the following order: EDUCATION, SERVICE, AUTO, HEALTH, WEB, FINANCE, OIL
salaries decreased as the distance from the metropolis increased and salaries increased as years of experience increased
companyid- No discernable trend
major - salaries increase in the following order: NONE, LITERATURE, BIOLOGY, CHEMISTRY, PHYSICS, COMPSCI, MATH, BUSINESS, ENGINEERING 
The graph of majors and industry(shown below) provided some interesting insights on closer look. We clearly see that NONE major have the lowest salary in any industry. We also see that the general trend above does not apply. For example even though generally Literature majors have the second lowest average salary, they have the highest average salary for the Education industry, and Chemistry and Biology majors have the two highest average salaries in the Health industry. So we can conclude that one's major will have a an impact on salary earned in any particular

## 3 Preprocessing for Modelling (file: Prepoc_salary_pred.ipynb) 
Using insights from EDA, the features with object data types were ordinally econded. Though we did not observe any particular variation of companyId with average salaries, we have encoded it as well to see its effect on our model data. In addition two features were engineered from yearsofExperience and milesFromMetropolis features. The yearsofExperience values were binned into six groups(range is 1 to 24 years) and the milesFromMetropolis binned into 10 groups (range is 1 to 99). The table below shows snipet of the data with the newly egineered features. The ordinal ecoder object of each feature is also saved in a dictionary for use on the test data set.


## 4. Modelling (file: Models_salary_pred.ipynb) 
Linear regression models were explored as a base model.The graph below shows the peformance of the several linear regression models of order 1,2 and 3. As the graph shows there is significant improvement in MSE from the first order model to the second order model(397 to 369), but much less improvement to the third order model(367). Higher order models(not shown in the graph) resulted in even less improvement in MSE at a much larger computational cost. A portion of the data (20%) was reserved for testing, and the MSE is based on the difference between the predicted salaries of the test data compared with actual salaries     
A decision tree model and three ensemble models (random forest, gradient boosting and XGBoost) were then explored as improvement on the base model. The hyperperameters of each model were tuned via a gridsearch to obtain optimum values. A 5-fold cross validatation was done to validate each model and results in each fold  was consistent. The data was then split into a train/test split (80/20) and the MSE of each model on the test data was obtained. The Decision tree model had the worst performance with an MSE of 388, the random forest model had an MSE of 364, the gradient boost had an MSE of 356 and the xtreme gradient boost model had an MSE of 355,  marginally better than the gradient boost model.The xtreme gradient boost model was therefore selected as the final model and the importance of each feature in prediction with the model is shown in the graph. 
During EDA we talked about dupicate feature values with different salaries(around 15000). We tried removing the duplicates including the 'originals' but there was no improvement in the MSE of the model. 


## 5. Deploy (file: Deploy_salary_pred.ipynb)
A script was written which automates the cleaning, prepocessing and modelling of the train data. With the train feature file path and the target file path as its input argument, the function converts the data set into pandas dataframes, cleans the data, checks for missing values and removes them, check for case uniformity and corrects if non-uniformity is detected, merges the features and target data on the the jobId column  checks, removes duplicates and rows with zero salary values. The data feature values are then ordinally encoded in additional columns and the encoder objects are saved in a dictionary. The entire data is then trained on the xgboost model and an object of the model is saved.
A second script which takes the file path of the test data set as it input argument was written. The function converts the data into a pandas dataframe, checks for case uniformity and corrects if necessary, and then removes duplicates after droping the jobId column. Using the saved ordinal encoder objects, each feature is encoded in a new column and the saved xgboost model is then use to predict the salaries of the test data. The predicted salaries is added to test dataframe which is saved as a csv file.


# Additions
A module (helper.py) which contains various functions written for each of the step above was created. At the beginning of each notebook, the helper module is imported and the required functions are invoked

## Areas to Improve on:
1. During EDA over 15000 rows were found with duplicate features but different salaries. This needs further analysis probably using KNN to select more accurate values
2. Engineering new features to Improve  MSE of model
3. Use objected oriented programming to make code more professional
