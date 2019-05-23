import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support,accuracy_score#,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import sklearn.grid_search as gs
import scipy.special

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
    t0=time.time()
    rfe.fit(dataset.X, dataset.y)
    old_dim=dataset.dim()
    t1=time.time()

    dataset.X= rfe.transform(dataset.X)
    print("Old dim %d New dim %d time: %0.4f)" % (old_dim,dataset.dim(),(t1-t0) ))
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

def compute_score(y_true,y_pred,as_str=True):
    precision,recall,f1,support=precision_recall_fscore_support(y_true,y_pred,average='weighted')
    accuracy=accuracy_score(y_true,y_pred)
    if(as_str):
        return "%0.4f,%0.4f,%0.4f,%0.4f" % (accuracy,precision,recall,f1)
    else:
        return (accuracy,precision,recall,f1)

def kl_div_matrix(dist_matrix,trans=False):
    if(trans):
        dist_matrix=dist_matrix.T
    def kl_helper(x_i,x_j):
        kl_array=scipy.special.kl_div(x_i,x_j)
        kl_array[kl_array==np.inf]=0.0
        return np.mean(kl_array)
    kl_matrix=[[ kl_helper(x_i,x_j)
                    for x_j in dist_matrix]
                        for x_i in dist_matrix]
    kl_matrix=np.around(kl_matrix,2)
    heat_map(kl_matrix)
    return kl_matrix

def heat_map(conf_matrix):
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')
    plt.show()

def show_stats(indiv):
    stats=(np.amin(indiv),np.median(indiv),np.mean(indiv),np.amax(indiv))
    print("min:%f median:%f mean:%f max:%f" %  stats)