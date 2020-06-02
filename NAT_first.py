import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

##Import Breast Cancer Wisconsin dataset
name = "NATv4_Base_Jf_elim.csv"
df = pd.read_csv(name)

#df = pd.DataFrame(df)

#df.head()
#print(whut)
X = df.iloc[:, 6:17] #features vectors
y = df.iloc[:, 18]  #class labels: 2 = benign, 4 = malignant
#print(y)
#print(X)
print(Counter(y))



le = LabelEncoder() #positive class = 1 (benign), negative class = 0 (malignant)
y = le.fit_transform(y)

#Replace missing feature values with mean feature value
X = X.replace('?', np.nan)
imr = SimpleImputer( missing_values= np.nan,strategy = 'mean')
imr = imr.fit(X)
X_imputed = imr.transform(X.values)

"""
df_class_0 = df[y == 0]
df_class_1 = df[y == 1]

df_class_0_under = df_class_0.sample()
"""
#Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size = 0.2, random_state = 42)


versample = SMOTE(random_state=42, k_neighbors=2, sampling_strategy=0.2)
versample2 = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
# fit and apply the transform
X_over, y_over = versample.fit_resample(X_train, y_train)

print(Counter(y_over))
X_train, y_train= versample2.fit_resample(X_over, y_over)

print(Counter(y_train))


categories = [col for col in X.columns]
# Variable importance
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print ("Features sorted by their score:")
z_train = categories
print( sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), z_train), reverse=True))
HNG = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), z_train), reverse=True)
# all equally likely covariance is same
#[(0.3523, 'satisfaction_level'), (0.1738, 'time_spend_company'), (0.1705, '
#print(HNG)
#HNG = HNG.tolist()

pipe_LinR = Pipeline([('scl', MinMaxScaler()),
			('clf', LinearRegression())])

pipe_lr = Pipeline([('scl', MinMaxScaler()),
			('clf', LogisticRegression(random_state=42))])
"""
pipe_lr_pca = Pipeline([('scl', MinMaxScaler()),
			('pca', PCA(n_components=2)),
			('clf', LogisticRegression(random_state=42))])
"""
pipe_rf = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestClassifier(random_state=42))])
"""
pipe_rf_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=2)),
			('clf', RandomForestClassifier(random_state=42))])
"""
pipe_svm = Pipeline([('scl', MinMaxScaler()),
			('clf', svm.SVC(random_state=42))])
"""
pipe_svm_pca = Pipeline([('scl', MinMaxScaler()),
			('pca', PCA(n_components=2)),
			('clf', svm.SVC(random_state=42))])
"""
pipe_NB = Pipeline([('scl', StandardScaler()),
			('clf', GaussianNB())])

pipe_KNN = Pipeline([('scl', StandardScaler()),
			('clf', KNeighborsClassifier())])
#pipe_KNN = Pipeline([('scl', StandardScaler()), ('clf', K)])

param_range = np.linspace(0,10,20)
param_range_2 = [i for i in range(20)]
d = np.linspace(0,1,9)
d = d.tolist()
param_range = param_range.tolist()

#grid_params_LinR = None#[{'clf__fit_intercept': ['True', 'False']}]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
		'clf__C': param_range,
		'clf__solver': ['liblinear']}] 

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],'clf__n_estimators':param_range_2}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
		'clf__C':  [0.1,0.2,0.5,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,20,25]}]

grid_params_NB = [{'clf__var_smoothing': d}]

grid_params_KNN = [{'clf__n_neighbors': param_range_2}]

# Construct grid searches
jobs = -1
sc = 'recall'
gs_lr = GridSearchCV(estimator=pipe_lr,
			param_grid=grid_params_lr,
			scoring=sc,
			cv=11) 
# test commentary


#gs_LinR = GridSearchCV(estimator=pipe_LinR,
#			param_grid=grid_params_LinR,
#			scoring='accuracy',
#			cv=10) 
"""			
gs_lr_pca = GridSearchCV(estimator=pipe_lr_pca,
			param_grid=grid_params_lr,
			scoring='accuracy',
			cv=10)
"""	
gs_rf = GridSearchCV(estimator=pipe_rf,
			param_grid=grid_params_rf,
			scoring=sc,
			cv=11, 
			n_jobs=jobs)
"""
gs_rf_pca = GridSearchCV(estimator=pipe_rf_pca,
			param_grid=grid_params_rf,
			scoring='accuracy',
			cv=10, 
			n_jobs=jobs)
"""
gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring=sc,
			cv=11,
			n_jobs=jobs)
"""
gs_svm_pca = GridSearchCV(estimator=pipe_svm_pca,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=10,
			n_jobs=jobs)
"""

gs_NB= GridSearchCV(estimator=pipe_NB,
			param_grid=grid_params_NB,
			scoring=sc,
			cv=11,
			n_jobs=jobs)

gs_KNN = GridSearchCV(estimator=pipe_KNN,
			param_grid=grid_params_KNN,
			scoring=sc,
			cv=11,
			n_jobs=jobs)
# List of pipelines for ease of iteration
grids = [gs_lr, gs_rf ,gs_svm, gs_NB, gs_KNN]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Logistic Regression',  1:'Random Forest',
		2: 'Support Vector Machine', 3: 'Naive Bayes', 4:'KNN'}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])	
    # Fit grid search	
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred)) 
    print('Test set precision score for best params: %.3f ' % precision_score(y_test, y_pred))

    print('Test set recall score for best params: %.3f ' % recall_score(y_test, y_pred))

    print('Test set f1-score for best params: %.3f ' % f1_score(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    data = confusion_matrix(y_test,y_pred)

    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
    plt.show()

    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

# Save best grid search pipeline to file
dump_file = 'best_gs_pipeline.pkl'
joblib.dump(best_gs, dump_file, compress=1)
print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))

