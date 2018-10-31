import numpy as np
import basic,utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from collections import Counter

def learning(handcrafted_path=None,deep_path=None):
    datasets_dict=get_datasets(handcrafted_path,deep_path)
    datasets=preproc_dataset(datasets_dict)
    y_true,all_pred=get_prediction(datasets)
    y_pred=vote(all_pred)
    return [y_j==y_i for y_i,y_j in zip(y_true,y_pred)]

def get_datasets(handcrafted_path=None,deep_path=None):
    if(not handcrafted_path and not deep_path):
        raise Exception("No dataset paths")
    handcrafted_dataset,deep_datasets=None,None
    if(handcrafted_path):
        handcrafted_dataset=basic.read_dataset(handcrafted_path)
    if(deep_path):
        deep_paths=utils.bottom_files(deep_path)
        if(len(deep_paths)==0):
            raise Exception("No datasets at " + deep_paths)
        deep_datasets=[ basic.read_dataset(path_i) for path_i in deep_paths]
	return {"handcrafted":handcrafted_dataset,"deep":deep_datasets}

def preproc_dataset(datasets_dict, hc_feats=250,deep_feats=100):
    datasets=datasets_dict["deep"]
    if(datasets_dict["handcrafted"]):
    	hc_data=datasets_dict["handcrafted"]
        datasets=[basic.unify_datasets([hc_data,deep_i])
                    for deep_i in datasets]
    for dataset_i in datasets:
        dataset_i.norm()
    return datasets
   
def get_prediction(datasets):
    result=[ train_model(i,data_i) 
                    for i,data_i in enumerate(datasets)]
    y_true=result[0][0]
    all_pred=[result_i[1] for result_i in result]
    return y_true,all_pred

def train_model(i,dataset_i):
    print("dataset %d" % i)
    train,test=dataset_i.split()
    clf=LogisticRegression()
    clf = clf.fit(train.X, train.y)
    y_pred = clf.predict(test.X)
    return test.y,y_pred

def vote(all_votes):
    all_votes=np.array(all_votes)
    y_pred=[]
    for vote_i in all_votes.T:
        count =Counter(vote_i)
        cat_i=count.most_common()[0][0]
        y_pred.append(cat_i)
    return y_pred 