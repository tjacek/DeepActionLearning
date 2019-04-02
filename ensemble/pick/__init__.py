import numpy as np
import ensemble.data,ensemble.tools,basic,utils
import ensemble.outliner
import plot

class ClfStats(object):
    def __init__(self, clf_quality=None,inspect_acc=None):
        if(not clf_quality):
            clf_quality=diagonal_criterion
        self.clf_quality = clf_quality
        if(not inspect_acc):
            inspect_acc=correlation_acc
        self.inspect_acc=inspect_acc

    def __call__(self,clf_acc,dict_arg,detector_path):
        inliners_matrix=feats_inliners(dict_arg,detector_path)
        feats_quality=self.clf_quality(inliners_matrix)
        return self.inspect_acc(clf_acc,feats_quality)

def mean_cat(quality):
    return np.mean(quality,axis=1)

def l2_mean_cat(quality):
    weights=np.mean(quality,axis=0)
    weights[weights>0.5]=0.0
    weights[weights>0]=1.0
    return weights*np.mean(quality,axis=1)

def diagonal_criterion(quality):
    diag=np.diagonal(quality)
    diag[diag<1.0]==0
    return diag

def correlation_acc(clf_acc,feats_quality):
    X=np.stack([clf_acc,feats_quality])
    return np.corrcoef(X)[0][1]

def resiudals(clf_acc,feats_quality):
    #regr=utils.linear_reg(clf_acc,feats_quality)
    regr,pred_acc=plot.show_regres(feats_quality,clf_acc)
    #pred_acc=regr.predict(feats_quality)
    return np.mean(np.abs( clf_acc-pred_acc))

def clf_quality(dict_arg,detector_path,quality_metric=None):
    if(not quality_metric):
        quality_metric=diagonal_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    return quality_metric(inliners_matrix)

def feats_inliners(dict_arg,detector_path):
    datasets,n_feats=ensemble.data.get_datasets(dict_arg,None,None)
    train_data=[dataset_i.split()[0] 
                    for dataset_i in datasets]
    detectors=ensemble.outliner.read_detectors(detector_path)
    n_cats=len(datasets)
    quality=[[ detectors.cat_separation(i,train_i,cat_j+1)
                for i,train_i in enumerate(train_data)]
                    for cat_j in range(n_cats)]
    return np.array(quality)

#def select_feats(in_path,out_path,n_feats=100):
#    deep_paths=utils.bottom_files(in_path)
#    deep_datasets=[ basic.read_dataset(path_i) for path_i in deep_paths]
#    for deep_i in deep_datasets:
#    	deep_i.norm()
#    deep_datasets=[ensemble.tools.rfe_selection(deep_i,n=n_feats) 
#                        for deep_i in deep_datasets]
#    utils.make_dir(out_path)
#    for i,data_i in enumerate(deep_datasets):
#        out_i=  out_path+"/"+deep_paths[i].split("/")[-1]
#        data_i.save(out_i)	