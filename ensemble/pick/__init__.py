import numpy as np
import ensemble.data,ensemble.tools,basic,utils
import ensemble.pick.outliner,ensemble.pick
import plot

class WeightedCrit(object):
    def __init__(self, crit,weights):
        if(type(weights)==list):
            weights=np.array(weights)   
        self.crit = crit
        self.weights=weights

    def __call__(self,raw_quality):
        quality=[self.weights*raw_i for raw_i in raw_quality]
        return self.crit(quality)
        
def mean_cat(quality):
    #quality[quality<1.0]=0.0
    return np.mean(quality,axis=1)

def w_mean_cat(quality):    
    weights=(np.amax(quality,axis=0)-np.amin(quality,axis=0))
    return np.mean(quality,axis=1) *weights
    #return np.median(quality,axis=1)

def diagonal_criterion(quality):
    diag=np.diagonal(quality)
    diag[diag<1.0]==0
    return diag

def get_cls_selector(dict_arg,detector_path,metric="weight",n_cls=3):
    if(not metric):
        metric=mean_cat
    if(metric=="weight"):
        metric=w_mean_cat
    quality=clf_quality(dict_arg,detector_path,quality_metric=metric)
    return ensemble.pick.selectors.sort_list(quality,n_cls)

def clf_quality(dict_arg,detector_path,quality_metric=None):
    if(not quality_metric):
        quality_metric=diagonal_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    return quality_metric(inliners_matrix)

def save_outliner_matrix(dict_arg,detector_path,out_path):
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    utils.save_matrix(out_path,inliners_matrix)

def feats_inliners(dict_arg,detector_path):
    datasets,n_feats=ensemble.data.get_datasets(**dict_arg)#,None,None)
    train_data=[dataset_i.split()[0] 
                    for dataset_i in datasets]
    detectors=ensemble.pick.outliner.read_detectors(detector_path)
    n_cats=len(datasets)
    quality=[[ detectors.cat_separation(i,train_i,cat_j+1)
                for i,train_i in enumerate(train_data)]
                    for cat_j in range(n_cats)]
    return np.array(quality)