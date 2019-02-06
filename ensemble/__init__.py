import numpy as np
from collections import Counter
import ensemble.data
import ensemble.tools,ensemble.inspect

class Ensemble(object):
    def __init__(self,clf=None):
        self.clf=clf if(clf) else tools.logistic_cls

    def get_prediction(self,datasets):
        result=[ self.train_model(i,data_i) 
                    for i,data_i in enumerate(datasets)]
        y_true=result[0][0]
        all_pred=[result_i[1] for result_i in result]
        return y_true,all_pred

    def train_model(self,i,dataset_i):
        clf,clf_name=self.clf()
        print("dataset %d %s" % (i,clf_name))
        train,test=dataset_i.split()
        clf = clf.fit(train.X, train.y)
        y_pred = clf.predict(test.X)
        return test.y,y_pred

def learning(handcrafted_path=None,deep_path=None,feats=(250,100)):
    datasets=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    ens=Ensemble(tools.SVC_cls)
    y_true,all_pred=ens.get_prediction(datasets)
    indiv=ensemble.inspect.cls_accuracy(y_true,all_pred,stats=False)
    y_pred=vote(all_pred)
    ensemble.tools.show_result(y_true,y_pred,datasets[0])

def vote(all_votes):
    all_votes=np.array(all_votes)
    y_pred=[]
    for vote_i in all_votes.T:
        count =Counter(vote_i)
        cat_i=count.most_common()[0][0]
        y_pred.append(cat_i)
    return y_pred