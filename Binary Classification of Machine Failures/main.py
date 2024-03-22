### create venv, activate, install relevant packages
### create imports


### read in input files

### routine initial inspection/exploration
    # training.into()       -- to check data types (strings, integers, etc.) and count entries
    # training.describe()   -- for all numeric data
    # training.columns()    -- quick inspection on all column names


### Feature engineering
    # to modify table to include columns that might impact results 
        # this might involve breaking down raw data into more decipherable data, e.g. instead of full cabin number, we can just use the letter category of the cabins
    # maybe normalize values (uh...i think take logs?)

### Pre-processing for models
    # fill in empty data(?) we have 3 options :
        # 1) drop columns with empty data
            #    cols_with_missing = [col for col in X_train.columns
            #         if X_train[col].isnull().any()]
        # 2) imputation : substitute empty numerical data with the mean/median
        # 3) imputation extended : substitute empty numerical data with mean/median, and create new column "Data MissiN?" with True for entries with empty original data
    # only inclde relevant data for specific models
    # basically data transformation
    # replace null with mean if normally distributed, if not, use median (in theory)

### define function to measure quality of each approach
    # Function for comparing different approaches, for example :
    #       def score_dataset(X_train, X_valid, y_train, y_valid):
    #           model = RandomForestRegressor(n_estimators=10, random_state=0)
    #           model.fit(X_train, y_train)
    #           preds = model.predict(X_valid)
    #           return mean_absolute_error(y_valid, preds)

import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score
    # MAE is a straightforward way to evaluate accuracy of model
from sklearn.model_selection import train_test_split
    # splits raw data into training and testing set
    # something like -- \
from sklearn.preprocessing import OrdinalEncoder
    # categories in the single column is now encoded with numbers. e.g. every day = 3, most days = 2, rarely = 1, never = 0
from sklearn.preprocessing import OneHotEncoder
    # categories in a single column are now in multiple columns. the entires would be binary, to indicate whether or not a row is that or not.
    # e.g. every day, most days, rarely, and never will be separated into 4 columns. if the row is rarely, there'll be a 1 in the rarely column, and 0 everywhere else
training = pd.read_csv(r'C:\Users\user\Python Stuff\Kaggle Projects\Wild Blueberry Yield\input\train.csv')
test = pd.read_csv(r'C:\Users\user\Python Stuff\Kaggle Projects\Wild Blueberry Yield\input\test.csv')

y = training['yield']
features = training.columns[0:-1]

X = training[features]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# training_info = training.describe()
# training_info.to_csv(r"C:\Users\user\Python Stuff\Kaggle Projects\Wild Blueberry Yield\work in progress\training_info.csv")

model = RandomForestRegressor(random_state = 3)
model.fit(train_X,train_y)
prediction = model.predict(val_X)

print('area under ROC is : ')
print(roc_auc_score(val_y, prediction))



final_prediction = model.predict(test)

output = pd.DataFrame({'Id': test.id,
                       'yield': final_prediction})

output.to_csv(r"C:\Users\user\Python Stuff\Kaggle Projects\Wild Blueberry Yield\work in progress\submission.csv", index=False)