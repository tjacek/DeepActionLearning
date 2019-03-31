import numpy as np
import sklearn
import ensemble.data,ensemble.tools,basic,utils
import ensemble.outliner

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

class CatSelector(object):
    def __init__(self, allowed_set):
        allowed_set=Set(allowed_set)
        self.selector = lambda inst_i: (inst_i.cat in allowed_set)

    def __call__(self,datasets):
        datasets=[ data_i.split(self.selector)[0] for data_i in datasets]
        return [ data_i.integer_labels() for data_i in datasets]

class DatasetSelector(object):
    def __init__(self,allowed_list):
        self.allowed_list=allowed_list

    def __call__(self,datasets):
        return [datasets[i] for i in self.allowed_list]

def make_data_selector(dict_arg,detector_path,
                        clf_critterion=None,selection_type=15):
    if(not clf_critterion):
        clf_critterion=diagonal_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    raw_quality=clf_critterion(inliners_matrix)
    if(type(selection_type)==int):
        allowed_list=np.argsort(raw_quality)[selection_type:]
    else:
        allowed_list=[ i for i,bool_i in enumerate(raw_quality)
                        if(bool_i==1.0)]
    return DatasetSelector(allowed_list)

def mean_cat(quality):
    return np.mean(quality,axis=1)

def diagonal_criterion(quality):
    diag=np.diagonal(quality)
    diag[diag<1.0]==0
    return diag

def correlation_acc(clf_acc,feats_quality):
    X=np.stack([clf_acc,feats_quality])
    return np.corrcoef(X)[0][1]

def resiudals(clf_acc,feats_quality):
    clf_acc,feats_quality=np.array(clf_acc), np.array(feats_quality)
    regr=sklearn.linear_model.LinearRegression()
    feats_quality=feats_quality[:,np.newaxis]
    regr.fit(feats_quality,clf_acc)
    pred_acc=regr.predict(feats_quality)
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