import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import sklearn.grid_search as gs

def SVC_cls():
    params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
    clf = gs.GridSearchCV(SVC(C=1,probability=True),params, cv=5,scoring='accuracy')
    return clf,"SVC"

def logistic_cls():
    return LogisticRegression(),"Logistic Regression"

def rfe_selection(dataset,n=100):
    if( (dataset is None) or dataset.dim()<n or n==0):
        return dataset
    svc = SVC(kernel='linear',C=1)
    rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
    rfe.fit(dataset.X, dataset.y)
    dataset.X= rfe.transform(dataset.X)
    print("New dim: ")
    print(dataset.X.shape)
    return dataset  

def show_result(y_true,y_pred,dataset=None):
    print(classification_report(y_true, y_pred,digits=4))
    print("Accuracy %f " % accuracy_score(y_true,y_pred))
    if(dataset):
        train,test=dataset.split()
        show_errors(y_true,y_pred,test.names)
    cf=confusion_matrix(y_true, y_pred)
    cf_matrix=pd.DataFrame(cf,index=range(cf.shape[0]))
    heat_map(cf_matrix)
    return cf_matrix
    
def show_errors(y_true,y_pred,names):
    errors=[ true_i!=pred_i for true_i,pred_i in zip(y_true,y_pred)]
    error_descs=[(name_i,y_pred[i])
                    for i,name_i in enumerate(names)
                        if(errors[i])]
    print(error_descs)

def compute_score(y_true,y_pred):
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,average='micro')
    recall=recall_score(y_true,y_pred,average='micro')
    f1=f1_score(y_true,y_pred,average='micro')
    return "%0.4f,%0.4f,%0.4f,%0.4f" % (accuracy,precision,recall,f1)

def heat_map(conf_matrix):
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')
    plt.show()

def show_stats(indiv):
    stats=(np.amin(indiv),np.median(indiv),np.mean(indiv),np.amax(indiv))
    print("min:%f median:%f mean:%f max:%f" %  stats)