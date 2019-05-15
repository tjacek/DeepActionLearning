import numpy as np
import matplotlib.pyplot as plt
import ensemble,ensemble.pick
from sklearn.metrics import accuracy_score

class SelectionExper(object):
    def __init__(self,clf,dict_arg,detector_path,metric=None):
        if(not metric):
            metric=ensemble.pick.mean_cat
        self.clf=clf
        self.quality_args={ 'dict_arg':dict_arg, 'detector_path':detector_path,
                            'quality_metric':metric}
    
    def __call__(self,ens_arg):#,n_datasets=20):
        ens=ensemble.Ensemble(clf=self.clf)
        datasets,n_feats=ens.get_datasets(**ens_arg)
        y_true,all_pred=ens.get_prediction(datasets)
        n_cats=len(all_pred)
        clf_indexs=self.eval_clfs(n_cats)
        sorted_preds=[ all_pred[clf_i] for clf_i in clf_indexs]
        print(clf_indexs)
        x=range(n_cats+1)[1:]
        def acc_helper(x_i):
            new_pred_i=ensemble.vote(sorted_preds[:x_i])
            return accuracy_score(y_true,new_pred_i)
        y=[ acc_helper(x_i) for x_i in x ]
        plot_accuracy(x,y)
        return y
    
    def eval_clfs(self,n_cats):
        quality=ensemble.pick.clf_quality(**self.quality_args)
        print(quality)
        clf_indexes=ensemble.pick.selectors.sort_list(quality,n_cats)
        return np.flip(clf_indexes)

def plot_accuracy(x,y):
    plt.plot(x,y)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of classifiers")
    plt.show()