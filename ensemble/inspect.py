import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import ensemble

def cls_quality(handcrafted_path,deep_path,feats=(250,100)):
    datasets_dict=ensemble.get_datasets(handcrafted_path,deep_path)
    ensemble.feat_reduction(datasets_dict,hc_feats=feats[0],deep_feats=feats[1])
    datasets=ensemble.preproc_dataset(datasets_dict)
    datasets=[ dataset_i.split()[0] for dataset_i in datasets]
    accuracy=[pred_accuracy(dataset_i) for dataset_i in datasets]
    print(accuracy)
    print(np.argsort(accuracy))

def pred_accuracy(dataset_i):
    def person_helper(person_j):
        selector_j=lambda inst_k: inst_k.person!=person_j
        train,test=dataset_i.split(selector_j)
        clf=LogisticRegression()
        clf = clf.fit(train.X, train.y)
        y_pred = clf.predict(test.X)
        return accuracy_score(test.y,y_pred)
    test_persons=dataset_i.unique_persons()
    print(test_persons)
    quality=[ person_helper(person_j) 
                for person_j in test_persons]
    print(quality)
    return np.amin(quality)

def cls_accuracy(y_true,all_pred,show=True,stats=True):
    indvidual_acc=[accuracy_score(y_true,pred_i)
                    for pred_i in all_pred]
    if(show):
        for i,indiv_i in enumerate(indvidual_acc):
    	    print("%d %f" % (i,indiv_i))
    if(stats):
        return basic_stats(indvidual_acc)
    else:
        return indvidual_acc

def basic_stats(array):
    return np.amin(array),np.median(array),np.mean(array),np.amax(array)