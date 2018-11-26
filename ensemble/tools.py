import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn import tree

def train_model(i,dataset_i,tree_cls=False):
    cls_name= "tree cls " if(tree_cls) else "logistic regression"
    print("dataset %d %s" % (i,cls_name))
    train,test=dataset_i.split()
    if(tree_cls):
        clf=tree.DecisionTreeClassifier()
    else:    
        clf=LogisticRegression()
    clf = clf.fit(train.X, train.y)
    y_pred = clf.predict(test.X)
    return test.y,y_pred

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

def show_result(y_true,y_pred):
    print(classification_report(y_true, y_pred,digits=4))
    print("Accuracy %f " % accuracy_score(y_true,y_pred))
    cf=confusion_matrix(y_true, y_pred)
    cf_matrix=pd.DataFrame(cf,index=range(cf.shape[0]))
    heat_map(cf_matrix)

def heat_map(conf_matrix):
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')
    plt.show()