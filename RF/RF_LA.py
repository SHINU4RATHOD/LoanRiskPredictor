# importing all required lib and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer


from sklearn.model_selection import train_test_split,cross_validate,cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sqlalchemy import create_engine
import joblib
import pickle
# load dataset and push dataset to mysql 
df = pd.read_json(r'C:/Users/SHINU RATHOD/Desktop/internship assignment/Recsify Technologies/03_dataset/Dataset/loan_approval_dataset.json') 
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
# df.to_sql('La', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from la;'
df = pd.read_sql_query(sql, engine)

df = df.drop(columns = ['Id','Married/Single', 'CITY', 'STATE'], axis = 1)


# 1. univariate analysis
df.info()
df.describe()
df.isnull().sum()

df.columns
df['Risk_Flag'].unique()
df['Risk_Flag'].value_counts()

# Histogram
plt.hist(df['Risk_Flag'], bins=5, color='skyblue', edgecolor='black')
plt.title('Histogram of Downtime')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

################# Extracting the independent and dependent var
x = df.drop('Risk_Flag', axis = 1)
y = df['Risk_Flag']   
# extracting numerical and categorical var
nf = x.select_dtypes(exclude = 'object').columns
cf = x.select_dtypes(include = 'object').columns


# 2. Bivariate analysis
 # Correlation matrix
corr_matrix = df[nf].corr()
print("Correlation Matrix:\n", corr_matrix)

# Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Scatter plot
df.columns
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Profession', y='Risk_Flag')
plt.title('Scatter Plot of Downtime vs. Torque')
plt.xlabel('Downtime')
plt.ylabel('Torque')
plt.grid(True)
plt.show()

# playing with AutoEDA Lib to check data quality
# 1) SweetViz
import sweetviz as sv
s = sv.analyze(df)
s.show_html()


# 2) D-Tale
import dtale 
d = dtale.show(df)
d.open_browser()
    

df.isnull().sum()
df.isnull().sum().sum() # 0 total N/A values
########################## creating the pipline for simpleImputer
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, nf)])
imputation = preprocessor.fit(x)
joblib.dump(imputation, 'meanimpute')

imputed_df = pd.DataFrame(imputation.transform(x), columns = nf)


# Defining a function to count outliers(counting outlier before applying winsorization and after winsorization tech) here there is no outlier present in this dataset
def count_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers
# Count outliers before Winsorization
outliers_before = imputed_df.apply(count_outliers)
outliers_before      # here 0 total num of outlier
outliers_before.sum()

imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()
############################## Define Winsorization pipeline
winsorizer_pipeline = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
X_winsorized = winsorizer_pipeline.fit_transform(imputed_df)
joblib.dump(winsorizer_pipeline, 'winsor')  

# Transform Winsorized data back to DataFrame
X_winsorized_df = pd.DataFrame(X_winsorized, columns=nf)

# Count outliers after Winsorization
outliers_after = X_winsorized_df.apply(count_outliers)
outliers_after

X_winsorized_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()

############################ creating pipline for MinmaxScaler(features scaling)
scale_pipeline = Pipeline([('scale', MinMaxScaler())])
X_scaled = scale_pipeline.fit(X_winsorized_df)
joblib.dump(X_scaled, 'minmax')

X_scaled_df = pd.DataFrame(X_scaled.transform(X_winsorized_df), columns = nf)
 


############################ creating pipline for OneHotEncoder
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(drop='first'))])
preprocess_pipeline = ColumnTransformer([('cat', encoding_pipeline, cf)])
X_encoded =  preprocess_pipeline.fit(x)   # Works with categorical features only
# Save the encoding model
joblib.dump(X_encoded, 'encoding')

encode_data = pd.DataFrame(X_encoded.transform(x).todense())
# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = X_encoded.get_feature_names_out(input_features = x.columns)
encode_data.info()

clean_data = pd.concat([X_scaled_df, encode_data], axis = 1)  # concatenated data will have new sequential index
clean_data.info() 

######## splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)

################################ Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf_Model = RandomForestClassifier()
rf_Model.fit(X_train, y_train)
y_train_pred = rf_Model.predict(X_train)
y_test_pred = rf_Model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)





# Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

# Create the parameter grid with the correct parameter name 'n_estimators'
param_grid = {
    'n_estimators': n_estimators,  
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
 
##################  Hyperparameter optimization with GridSearchCV
rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)
rf_Grid.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = rf_Grid.best_params_
cv_rf_grid = rf_Grid.best_estimator_

# Predictions on training and test sets
y_train_pred = cv_rf_grid.predict(X_train)
y_test_pred = cv_rf_grid.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))



################# Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose = 0, n_jobs = -1)

rf_Random.fit(X_train, y_train)
rf_Random.best_params_
cv_rf_random = rf_Random.best_estimator_

# Predictions on training and test sets
y_train_pred = cv_rf_random.predict(X_train)
y_test_pred = cv_rf_random.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))


########### Cross Validation implementation
def cross_validation(model, _X, _y, _cv = 5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model, X = _X, y = _y, cv = _cv, scoring = _scoring, return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })
Random_forest_result = cross_validation(cv_rf_random, X_train, y_train, 5)
Random_forest_result


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.1, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.1, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize = 30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.legend()
        plt.grid(True)
        plt.show()

model_name = "RandomForestClassifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Random_forest_result["Training Accuracy scores"],
            Random_forest_result["Validation Accuracy scores"])
