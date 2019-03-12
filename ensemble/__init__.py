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
        datasets,n_feats=self.get_datasets(handcrafted_path,deep_path,feats)
        y_true,all_pred=self.get_prediction(datasets)
        y_pred=vote(all_pred)
        return self.show_result(show,y_true,y_pred,datasets,n_feats)
    
    def get_datasets(self,handcrafted_path=None,deep_path=None,feats=(250,100)):
        datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
        if(self.selector):
            datasets=[ data_i.split(self.selector)[0] for data_i in datasets]
            datasets=[ data_i.integer_labels() for data_i in datasets]
        return datasets,n_feats

    def get_prediction(self,datasets):
        result=[ self.train_model(i,data_i) 
                    for i,data_i in enumerate(datasets)]
        y_true=result[0][0]
        all_pred=np.array([result_i[1] for result_i in result])
        return y_true,all_pred

    def show_result(self,show,y_true,y_pred,datasets,n_feats):
        if(show):
            cf_matrix=ensemble.tools.show_result(y_true,y_pred,datasets[0])
            if(type(show)==str):
                np.savetxt(show,cf_matrix.values,delimiter=",")
        else:
            as_str=not self.selector
            return ensemble.tools.compute_score(y_true, y_pred,as_str),n_feats

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

class WeightedEnsemble(Ensemble):
    def __init__(self,weights, clf=None,prob=False,selector=None):
        super(WeightedEnsemble, self).__init__(clf,prob,selector)
        self.weights=weights

    def __call__(self,handcrafted_path=None,deep_path=None,feats=(250,100),show=True):
        datasets,n_feats=self.get_datasets(handcrafted_path,deep_path,feats)
        y_true,votes,names_true=self.get_prediction(datasets)
        y_pred=self.count_votes(votes,names_true)
        return self.show_result(show,y_true,y_pred,datasets,n_feats)

    def get_prediction(self,datasets):
        train,test=zip(*[data_i.split() for data_i in datasets])
        y_true,names_true=test[0].y,test[0].names
        clfs,clf_name=zip(*[self.clf() for data_i in datasets])
        n_cats=len(clfs)
        for clf_i,train_i in zip(clfs,train):
            clf_i.fit(train_i.X, train_i.y)
        ens_data=data.to_ensemble_samples(test,split=False)
        votes={ name_j:[ clf_i.predict(sample_ij) 
                    for clf_i,sample_ij in zip(clfs,ens_sample_j)] 
                        for name_j,ens_sample_j in ens_data.items()}
        votes={ name_j.strip():to_vector_votes(vote_j,n_cats) 
                    for name_j,vote_j in votes.items()}
        return y_true,votes,names_true

    def count_votes(self,votes,names_true):
        def vote_helper(name_j):
            vote_j=votes[name_j]
            weights_j=self.weights[name_j]
            print(vote_j.shape)
            weigted_vote_j=vote_j#[ weights_ij*vote_ij 
                           #     for vote_ij,weights_ij in zip(vote_j,weights_j)]
            weigted_vote_j=np.sum(np.array(weigted_vote_j),axis=0)
            return np.argmax(weigted_vote_j) +1                  
        y_pred=[ vote_helper(name_j) for name_j in names_true]
        print(y_pred)
        return y_pred

def to_vector_votes(votes,n_cats):
    return np.array([utils.one_hot(cat_i-1,n_cats) 
                        for cat_i in votes])

def vote(raw_votes):
    sumed_votes=np.sum(raw_votes,axis=0)
    result=[np.argmax(vote_i)+1 for vote_i in sumed_votes]
    return result