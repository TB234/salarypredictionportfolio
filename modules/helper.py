import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor
import xgboost 


def upload_file_csv(file_link):
    #print(" Displaying nlot Snipet of Data")
    return  pd.read_csv(file_link)

def summary_stat(data):
    #print ('Shape of data is {} ' .format(data.shape), end ='\n\n')
    print('Data has %d rows by %d columns' %(data.shape[0], data.shape[1]), end ='\n\n')
    print ('Data type', data.dtypes, sep = '\n')
    return data.describe(include = 'all')


class summary_stats():
    def __init__(self, data):
        self.data = data
        #print('Data has %d rows by %d columns' %(data.shape[0], data.shape[1])
        print ('Shape of data is {} ' .format(data.shape), end ='\n\n')
        print ('  Data type', data.dtypes, sep = '\n')


def uniq_values_in_feature(data, feature):
#accept feature as string or list of strings and drop feature if all values in features are unique 
    print("")
    x =[]
    if type(feature) == list:
        for i in feature:
            uniq_vals = data[i].unique()
            l = len (uniq_vals)
            print('There are %s  unique values of %s' %(l, i) )
            #[print(val, end = ' ') for val in uniq_vals]
            if data.shape[0] == l:
               x.append(i)
            else: 
                continue
        return drop_feature(data, x)
    else:
        uniq_vals = data[feature].unique()
        l = len (uniq_vals)
        print('There are %s  unique values of %s' %(l, feature) )
        #[print(val, end = ',') for val in uniq_vals]


def data_set_joiner(data_1, data_2, joiner, How):
    return data_1.join(data_2.set_index(joiner), on = joiner, how = How )
    #return pd.merge(data_1, data_2, on = 'jobId')


def missing_vals(data):
    print('Percent missing of total')
    print(data.isnull().sum()/len(data))


def case_check(data):
    print('Checking if case in feature values are uniform')
    counter = data.shape[0]
    for i in data.columns:
        if data[i].dtype == object:
            total_uppercase = sum(data[i].apply(lambda x: x.isupper()))
            if counter == total_uppercase:
                print (i, '--->','All upppercase')
            elif total_uppercase == 0:
                print (i, '--->', 'All lowercase')
            else: 
                print (i, '--->', 'Mixed case')
                data[i] = data[i].str.upper()
                print(i, ' converted to uppercase')
    print('')             
    return data
             


def drop_missing (data, target): #drop missing value and empty strings and zero target values
    int_rows = data.shape[0]
    missing_vals(data)
    #print('Output After dropping missing values')
    pd.options.mode.use_inf_as_na = True #sets empty strings as nan
    data.dropna(inplace = True)
    if target in data.columns:
        print ('\n', 'Dropped %d rows with zero %s values' %(int_rows - data[data[target] > 0].shape[0], target))
        return data[data[target] > 0]
    else:
        return data

#checks for duplicates
def check_dup(data):
    duplicates = sum(data.duplicated(keep = 'first') == True)
    print('%d duplicates found' %duplicates)

#drops duplicates
def drop_dup(data):
    duplicates = sum(data.duplicated() == True)
    print('\n', '%d duplicates found and removed' %duplicates)
    if duplicates > 0:
        #print('Output after dropping duplicates')
        return data.drop_duplicates()
    else:
        return data

#drop features
def drop_feature(data, feature):
    print(feature, ' dropped')
    return data.drop( axis=1, columns = feature)


def group(data, features, fillter = 0):
    grouped_data = data.groupby(features).agg(list)
    if fillter == 0:   
        return grouped_data
    else:   
        return grouped_data[grouped_data['salary'].apply(lambda x: len(x)> fillter)]


def group_mean(data, features, target):
    grouped_data = data.groupby(features).mean().sort_values(target)
    if fillter == 0:   
        return grouped_data



#show duplicated data on features set with thier corresponding target
def dup_feat(data, target_col):
    print ('Number of duplicates found ---> ', sum(data.drop(columns = target_col).duplicated()== True))
    return data[data.drop(columns = target_col).duplicated()]


#function drops duplicates after removing target_col, and then merges back target_col. It will remove rows in target_col corresponding to duplicate rows in features data only
#target columns in string or list of strings
def drop_dup1(data, target_col = None ):    
    if target_col == None:
        print ('Number of duplicates found ---> ', sum(data.duplicated() == True))
        return data.drop_duplicates()
    else:
        print ('Number of duplicates found ---> ', sum(data.drop(columns = target_col).duplicated()== True)) 
        return data[data.drop(columns = target_col).duplicated()== False] 
    


#Fuction groups data using mean, median, mode of target for duplicated features
# types in string
def select_clean_met(data, feature_list, types):
    if types == 'mean':
         return data.groupby(feature_list).mean().reset_index()
    elif types == 'max':
        return data.groupby(feature_list).max().reset_index()
    elif types == 'min':
        return data.groupby(feature_list).min().reset_index()
    elif types== 'median':
        return data.groupby(feature_list).median().reset_index()


#function bins values of features into specified groups
def add_binned_feat(data, feature, num_bins, bin_labels, new_feat_name):
    data[new_feat_name] = pd.cut(data[feature], num_bins, labels = bin_labels ).astype('int64')
    print (feature, ' values binned')
    return data


#feature_list should be a string or list of strings
def drop_feats(data, feature_list):
    data.drop(columns = feature_list, inplace = True)
    return data


#convert features from one type to another
def conv_feat_type(data, feature, to_type ):
    for i in feature:
        if i.dtype != to_type:
            data[i] = data[i].astype(to_type)
            print(i, 'data type --->',  data[i].dtype)
        else:
            print(i, 'data type --->',  data[i].dtype)
    return data


#Divides data into train and test split. Need to import shuffle from sklearn.utils
def div_train_test(data, train_frac, seed):
    data = shuffle(data, random_state = seed)
    sample_train = data.sample(frac = train_frac, random_state = seed)
    sample_test = data.drop(sample_train.index)
    print ('data splitted into sample_train and sample_test')
    return sample_train, sample_test


# aggregates targets after grouping to create new features
def new_tar_feat(data, target):
    data['group_mean']= data[target].apply(lambda x: np.mean(x))
    data['group_max']= data[target].apply(lambda x: np.max(x))
    data['group_min']= data[target].apply(lambda x: np.min(x))
    data['group_std']= data[target].apply(lambda x: np.std(x))
    data['group_median']= data[target].apply(lambda x: np.median(x))
    data['group_range']= data[target].apply(lambda x: np.ptp(x))
    return data


#merger function
def merger(base_data, merger_data, merg_on_feat, how= None):
    return base_data.merge(merger_data, on = merg_on_feat, how = how )


#creating a pivot table and color map
def color_map(data, pivot_row, pivot_column):
    group = data[[pivot_row, pivot_column, 'salary']]
    group = group.groupby([pivot_row, pivot_column], as_index = False).mean()

    pivot_table = group.pivot(index = pivot_row, columns = pivot_column)

    fig1, ax = plt.subplots( figsize= (10,10))

    clr_map = ax.pcolormesh(pivot_table, cmap='RdBu')

    ax.set_xticks(np.arange(pivot_table.shape[1])+ 0.5)
    ax.set_yticks(np.arange(pivot_table.shape[0])+ 0.5)

    ax.set_xticklabels(pivot_table.columns.levels[1])
    ax.set_yticklabels(pivot_table.index)

    plt.xticks(rotation=90)


    #fig1.colorbar(clr_map)


#create pivot table
def pivot_table(data, pivot_row, pivot_column, target):
    group = data[[pivot_row, pivot_column, target]].groupby([pivot_row, pivot_column], as_index = False).mean()
    return group.pivot(index = pivot_row, columns = pivot_column).sort_values( data[pivot_row].unique()[0], axis = 1)


#plot line graphs of feature against salary with an additional feature dimension as filter 
def plot_feature_corr(data, feature1, feature2, target, order = 'y'):
    grouped_data = data[[feature1, feature2, target]].groupby([feature1, feature2], as_index = True).mean()
    grouped_data.reset_index(level = feature2, inplace = True)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0,0,1,1])
    
    if order == 'y':
        counter = 0
        for i in data[feature1].unique():
            try:
                ax.plot(grouped_data.loc[i].sort_values(target)[feature2], grouped_data.loc[i].sort_values(target)[target], 
                    color = 'C'+ str(counter), marker = 's', label=i )
            except:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )    
            counter += 1
            
    elif order == 'n':
        counter = 0
        for i in data[feature1].unique():
            try:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )
            except:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )    
            counter += 1
            
    ax.set_ylabel('Average Salary ')
    ax.set_xlabel(feature2)
    ax.legend(  loc = 'lower right',  )



#plot a categorical feature value against target 
def feat_val_graph(data, pivot_row, pivot_column, target, row_feat_val):
    table = pivot_table(data, pivot_row, pivot_column, target).loc[row_feat_val].reset_index()
    table.drop('level_0', axis = 1, inplace = True)
    table. rename(columns = {row_feat_val: target.upper()}, inplace = True)
    plt.figure(figsize = (5,5))
    plt.plot(table[pivot_column], table[target.upper()] )
    plt.title(label = row_feat_val +' ' + pivot_row.upper() )
    plt.ylabel( ylabel = 'Average' + ' ' + target.upper())
    plt.xlabel( xlabel = pivot_column.upper())


#Plot bar graphs of features
def feat_dist(data, feat_list):
    fig, ax = plt.subplots(len(feat_list),1, figsize = (20,20))
    cnt = 0
    for i in feat_list:
        ax[cnt].bar(data[i].sort_values().unique(), data[i].value_counts().sort_index())
        cnt += 1



#Plot box plots
def feat_boxplot(data, feature, target):
    fig = plt.figure( figsize=(10,10))
    axs = fig.add_axes([0,0,1,1])
    sns.boxplot(x = feature, y = target, data =data, ax = axs)


    
#label encoder function
#feature and enc_feat_name can be strings or list of strings 
def lab_enc(data, feature, enc_feat_nam):
    le = LabelEncoder()
    if type(feature) == list:
        cnt = 0  
        for i in feature:
            data[enc_feat_nam[cnt]] = le.fit_transform(data[i])
            cnt += 1
    elif type(feature) == str:
        data[enc_feat_nam] = le.fit_transform(data[feature])
    return data



#Ordinal encoder function
#category_order_list could be a single list or list of multiple lists
#feature, enc_feat_nam could be string or list of strings
def ord_enc(data, feature, cat_ord_lst, enc_feat_nam):
    ord_encoder = OrdinalEncoder(categories = cat_ord_lst, dtype= 'int64')
    if type(feature)== str:
        data[enc_feat_nam]= ord_encoder.fit_transform(data[[feature]])
    elif type(feature) == list:
        data.reset_index(inplace = True)
        data.drop(axis = 1, columns= 'index', inplace = True )
        data[enc_feat_nam] = pd.DataFrame(ord_encoder.fit_transform(data[feature]))   
    return data



#Ordinal encoder function #2 for train data
def cat_ord_enc(data, target, feature= None):
    if feature == None:
        feature = cat_feat_list(data, target)
    if type(feature)== str:
        ordered_data = data.groupby(feature).mean().sort_values(target).index.unique() 
        ord_encoder = OrdinalEncoder(categories = [ordered_data], dtype= 'int64')
        data[feature + '_cat']= ord_encoder.fit_transform(data[[feature]])
        data[feature + '_cat'] = data[feature + '_cat'] + 1
        
    elif type(feature) == list:
        encoder_dict = {}
        for i in feature:
            ordered_data = data.groupby(i).mean().sort_values(target).index.unique() 
            ord_encoder  = OrdinalEncoder(categories = [ordered_data], dtype= 'int64')
            data[i + '_cat'] = ord_encoder.fit_transform(data[[i]]) + 1
            encoder_dict[i] = ord_encoder
            
        joblib.dump(encoder_dict, 'features_ordinal_econders')

    print('\n', 'Data encoded')        
    return data


#ordinal encoder for test data
def test_ord_enc(data):
    enc_dict = joblib.load('features_ordinal_econders')
    feature = cat_feat_list(data, None)
    for i in feature:
        data[i + '_cat'] = enc_dict[i].fit_transform(data[[i]]) + 1
    print('Data encoded')
    return data 



# Function that assign numbers to the unique feature values and stores in dictionary 
def num_assign(data, feature):
    map_val = {}
    unique_val = data[feature].unique()
    
    for i in range(len(unique_val)):
        map_val[unique_val[i]] = i
    return map_val


#alternative to sklearn label or ordinal encoder
def map_encod(data, feat, lis, enc_feat_nam):
    if type(feat) == str:
        data[enc_feat_nam] = data[feat].map(lis)
    elif type(feat) == list:
        for i in range(len(feat)):
            data[enc_feat_nam[i]]= data[feat[i]].map(lis[i])
    return data


#One hot encode function
def one_hot_enc(data, feat):
    return pd.get_dummies(data, columns = feat, drop_first = True)

#function creates list of numerical features in dataframe
def num_feat_list(data, target):
    c= []
    for feature in data.columns:
        if data[feature].dtype != object and feature != target :
            c.append(feature)
    return c

#function creates list of features in dataframe
def feat_list(data, target):
    c= []
    for feature in data.columns:
        if feature != target :
            c.append(feature)
    return c

#function creates list of categorical features in dataframe
def cat_feat_list(data, target):
    c= []
    for feature in data.columns:
        if data[feature].dtype == object and feature != target :
            c.append(feature)
    return c


# Fuction determines train and test data split. If user provides a separate test sample it is used, otherwise a portion of train sample is reserved for testing
def use_data(train_sample, target, test_sample = 'auto', num_features = 'auto'):
    
    #gets all numerical features from data if auto is used
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)
    
    #Check
    if type(test_sample) == str:
        x = train_sample[ num_features ]
        y = train_sample [target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

    else:
        x_train = train_sample[ num_features ]
        y_train = train_sample [target]
        x_test = test_sample[num_features]
        y_test = test_sample[target]
        
    return x_train, y_train, x_test, y_test


#Feature importance plot
def feature_importance(model, num_features):
    feature_importances = []
    pos = np.arange(len(num_features)) + .5
    importance_percentage = []
    
    for i in range(len(num_features)):
        feature_importances.append(model.feature_importances_[i])
    
    for i in feature_importances:
        importance_percentage.append(i*100)
                  
    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(pos, importance_percentage, align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(num_features)
    ax.set_xlabel('Relative Importance %')
    plt.title('Feature Importance')
    plt.show()



#Linear regression model

def poly_reg(train_sample,  target, order,  num_features = 'auto', test_sample = 'auto'):

    lrm = LinearRegression()

    #gets all numerical features from data if auto is used
    #if num_features == 'auto':
     #           num_features = num_feat_list(train_sample, target = target)
            
    x_train, y_train, x_test, y_test = use_data(train_sample, target, test_sample = 'auto')

  
    mse_train = []
    mse_test = []
    for i in order:
        pr = PolynomialFeatures(degree=i)
        x_train_pr = pr.fit_transform(x_train)
        x_test_pr = pr.fit_transform(x_test)

        lrm.fit(x_train_pr, y_train)

        yhat_train = lrm.predict(x_train_pr)
        yhat_test = lrm.predict(x_test_pr)

        mse_train.append(mean_squared_error(y_train, yhat_train))
        mse_test.append(mean_squared_error(y_test, yhat_test))
      
    mse_train
    mse_test

    fig, (ax2) = plt.subplots(1,1, figsize=(10,10)) 

    ax2.plot(order, mse_train, color = 'red' , marker = '*', label = 'train' )

    ax2.plot(order, mse_test, color = 'green', marker = '*', label = 'test' )

    ax2.set_xlabel('Polynomial order')
    ax2.set_ylabel('MSE')
    ax2.legend(loc= 'lower left')

    plt.show



#========================
def xg_boost(train_sample,  target,  num_features = 'auto', test_sample = 'auto', estimator = None):
    
    if estimator == None:

        xg_bst = xgboost.XGBRegressor(n_jobs = -1, random_state = 2, n_estimators = 100, max_depth = 8, min_child_weight= 8, 
                           reg_lambda = 1, reg_alpha = 20, learning_rate = 0.09, base_score= 0.5, colsample_bytree = 0.8, subsample = 1)
    else: 
        xg_bst = estimator
           
    #gets all numerical features from data if auto is used
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)

    #trains on entire data if no test sample provided. Used to get final model        
    if test_sample == None:
        x_train = train_sample[ num_features ]
        y_train = train_sample [target]
        
        xg_bst.fit(x_train, y_train)
        yhat_train = xg_bst.predict(x_train)
        train_mse = mean_squared_error(yhat_train, y_train)
        print('\n', 'The mean squared error for entire train data ', train_mse)

        feature_importance(xg_bst, num_features)
        
        model_file_name = 'final_xgb_model_saved'
        joblib.dump(xg_bst, model_file_name )
        print ('final xgb model saved as --->', model_file_name)
        
        return xg_bst
        
    #splits train sample for train and test or uses test sample provided        
    else: 
        
        x_train, y_train, x_test, y_test = use_data(train_sample, test_sample, num_features )
        
        #commented because it takes long to run
        #xgb_cross = cross_validate(xg_bst, x_train, y_train, n_jobs= -1, scoring = ('neg_mean_squared_error'), 
                             # cv=5, return_train_score=True, return_estimator= False, verbose = 3)
        #print('Cross validation scores:', '\n', rf_cross, )

        xg_bst.fit(x_train, y_train)
        
        feature_importance(xg_bst, num_features)     

        yhat_train = xg_bst.predict(x_train)
        yhat_test = xg_bst.predict(x_test)
        
        train_mse = mean_squared_error(yhat_train, y_train)
        mse = mean_squared_error(yhat_test, y_test)
        
        print('\n','The train mean squared error', train_mse)
        print('The test mean squared error ', mse)
        
        fig = plt.figure()

        ax1=  sns.distplot(y_test, hist= False, color = 'red', label= 'Actual')

        sns.distplot(yhat_test, hist=False, color = 'blue', ax=ax1, label= 'Predicted')
        
        ax1.set_title(label = "TEST SAMPLE DISTRIBUTION")
        
        
        joblib.dump(xg_bst, 'saved_xgb_model' )

        
        return xg_bst
    
#===================

#Compares performance of 4 models- Decision Tree, Random Forest, Gradient boost, XGB
def best_model_mse(train_sample,  target,  num_features = 'auto', test_sample = 'auto'):
    
    #gets all numerical features from data if auto is used
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)
            
    x_train, y_train, x_test, y_test = use_data(train_sample, target, test_sample = 'auto')

#models
    dt = tree.DecisionTreeRegressor(max_depth= 30, max_features = 'auto', 
                                        min_samples_split=130, random_state = 1, ccp_alpha = 0.0001 )

    rf = RandomForestRegressor(n_estimators = 100, min_samples_split = 20, min_samples_leaf = 8, verbose = 0,
                               max_features = 'sqrt', max_samples= 100000, random_state = 1, oob_score= True, n_jobs = -1, bootstrap = True )

    grad_bst = GradientBoostingRegressor(n_estimators = 100, max_depth= 9, max_features = 5, learning_rate = 0.1, 
                                   min_samples_split = 8, min_samples_leaf = 8, verbose = 0, random_state = 3)


    xg_bst = xgboost.XGBRegressor(n_jobs = -1, random_state = 2, n_estimators = 100, max_depth = 8, min_child_weight= 8, 
                               reg_lambda = 1, reg_alpha = 20, learning_rate = 0.09, base_score= 0.5, colsample_bytree = 0.8, subsample = 1)


    models = {dt : 'Dec_tree', rf: 'rand_forest', grad_bst : 'gradient_boost', xg_bst : 'xtreme_grad_boost'}
    #models = {dt : 'Dec_tree', rf: 'rand_forest'}

    #print ('CROSS VAL MSE SCORES')

    #for keys in models.keys():
    #    cross = cross_validate(keys, x_train, y_train, n_jobs= -1, scoring = ('neg_mean_squared_error'),
    #                           cv=5, return_train_score=True, return_estimator= False, verbose = 0)
    #   print(models[keys], '----> ', cross['test_score'])

    model_mse = {}
    for keys in models.keys():
        keys.fit(x_train, y_train)

        yhat_test = keys.predict(x_test)

        mse = mean_squared_error(yhat_test, y_test)
        model_mse[models[keys]] = mse

    print('MODEL MSEs')
    for key,value in model_mse.items():
        print(key, '--->', value)
        
    best_mod = min(model_mse.keys(), key=(lambda k: model_mse[k]))
    
    for key, value in models.items():
        if value == best_mod:
            joblib.dump(key, 'saved_best_model')
            
            print ('\n', 'Best model ----> ', value, '  (saved as: saved_best_model)')
         
            
            feature_importance(key, num_features)
            
            return key


#cleans, processes raw train data and returns model for predicting
def combined_proc_modelling(train_features_link, train_target_link):
    
    train_features = upload_file_csv(train_features_link) #get train features
    train_target = upload_file_csv(train_target_link)     #get train target
    
    x = merger(train_features, train_target, 'jobId', how= 'left') #merge features and target data set
    
    y = case_check(x) #checks that cases in each feature are uniform and corrects otherwise
    
    z = drop_missing(y, 'salary') #drops rows with missing values and rows with target values less than or equal to 0 
    
    m = drop_dup(drop_feature(z, 'jobId' )) #further removes duplicate rows after droping jobId column
    
    r = cat_ord_enc(m, 'salary')   #endcodes categorical features
    
    b = add_binned_feat(r, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') #bins years experience and milesfromMetropolis feautures
    a = add_binned_feat(b, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') #average salary inversly propotional to milesFromMetropolis
    
    
    # trains entire processed data using xtreme gradeint boost algorithm to create final model that is saved   
    final_model = xg_boost(a,  'salary',  num_features = 'auto', test_sample = None, estimator = None) 
    
    return final_model


#predicts target given cleaned data and model for prediction
def model_predict(data, model):
   
    num_features = num_feat_list(data, 'salary')
    
    x_pred = data[ num_features ]
    
    y_pred = model.predict(x_pred)
    
    return np.rint(y_pred)


#clean and process data, predicts target and exports data with predictions as csv file
def process_predict_pipe(test_features_link):
    
    x = upload_file_csv(test_features_link)
    
    y = case_check(x) #checks that cases in each feature are uniform and corrects otherwise
    
    m = drop_feature(y, 'jobId' ) # droping jobId column
    
    r = test_ord_enc(m)   #endcodes categorical features
    
    b = add_binned_feat(r, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') #bins years experience and milesfromMetropolis feautures
    a = add_binned_feat(b, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') #order of range reversed because average salary inversly propotional to milesFromMetropolis
    
    model = joblib.load('final_xgb_model_saved')
    
    salary_predictions = model_predict(a, model)
    
    x['predicted_salary'] = salary_predictions
    
    output_file = 'predicted_salary.csv'
    
    x.to_csv(output_file)
    print('Predictions saved as csv to ---> ', output_file )
    
    return x
