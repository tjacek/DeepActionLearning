import numpy as np
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

def make_data_selector(dict_arg,detector_path,clf_selection=None):
    if(not clf_selection):
        clf_selection=mean_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    allowed_list=clf_selection(inliners_matrix)
    return DatasetSelector(allowed_list)

def diagonal_criterion(quality):
    diag=np.diagonal(quality)
    diag[diag<1.0]==0
    return diag

def correlation_acc(clf_acc,feats_quality):
    X=np.stack([clf_acc,feats_quality])
    return np.corrcoef(X)[0][1]




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

def diagonal_selection(quality):
    clf_quality=diagonal_criterion(quality)
    return [ i for i,bool_i in enumerate(clf_quality)
                if(bool_i==1.0)]



def mean_criterion(quality):
    clf_quality=np.mean(quality,axis=1)
    return np.argsort(clf_quality)[15:]
#def std_cat(quality):
#    cat_std=np.mean(quality,axis=0)
#    hardest_cat=np.argmax(cat_std)
#    return quality[:,hardest_cat]

def mean_cat(quality):
    print(np.amin( quality,axis=0))
    print(np.amin( quality,axis=1)  )
    return np.mean(quality,axis=1)

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