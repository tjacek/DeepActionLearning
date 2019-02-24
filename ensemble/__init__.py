import numpy as np
from collections import Counter
import ensemble.data
import ensemble.tools,ensemble.inspect
import utils

class Ensemble(object):
    def __init__(self,clf=None,prob=False,selector=None):
        self.clf=clf if(clf) else tools.logistic_cls
        self.prob=prob
        self.selector=selector

    def __call__(self,handcrafted_path=None,deep_path=None,feats=(250,100),show=True):
        datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
        if(self.selector):
            datasets=[ data_i.split(self.selector)[0] for data_i in datasets]
            datasets=[ data_i.integer_labels() for data_i in datasets]
        y_true,all_pred=self.get_prediction(datasets)
        y_pred=vote(all_pred)
        if(show):
            cf_matrix=ensemble.tools.show_result(y_true,y_pred,datasets[0])
            if(type(show)==str):
                np.savetxt(show,cf_matrix.values,delimiter=",")
        else:
            as_str=not self.selector
            return ensemble.tools.compute_score(y_true, y_pred,as_str),n_feats
        
    def get_prediction(self,datasets):
        result=[ self.train_model(i,data_i) 
                    for i,data_i in enumerate(datasets)]
        y_true=result[0][0]
        all_pred=np.array([result_i[1] for result_i in result])
        return y_true,all_pred

    def train_model(self,i,dataset_i):
        train,test=dataset_i.split()
        clf,clf_name=self.clf()
        print("dataset %d %s" % (i,clf_name))
        clf = clf.fit(train.X, train.y)
        n_cats=train.n_cats()
        if(self.prob):
            y_pred = clf.predict_proba(test.X)
        else:
            y_pred=clf.predict(test.X)
            y_pred=to_vector_votes(y_pred,n_cats)
            print(y_pred.shape)
        return test.y,y_pred

def to_vector_votes(votes,n_cats):
    return np.array([utils.one_hot(cat_i-1,n_cats) 
                        for cat_i in votes])

def vote(raw_votes):
    sumed_votes=np.sum(raw_votes,axis=0)
    result=[np.argmax(vote_i)+1 for vote_i in sumed_votes]
    return result