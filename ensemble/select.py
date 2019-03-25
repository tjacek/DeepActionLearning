import numpy as np
import ensemble.data,ensemble.tools,basic,utils
import ensemble.outliner

def acc_correlation(clf_acc,dict_arg,detector_path,quality_metric=None):
    if(not quality_metric):
        quality_metric=diagonal_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    feats_quality=quality_metric(inliners_matrix)
    X=np.stack([clf_acc,feats_quality])
    return np.corrcoef(X)[0][1]

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

def diagonal_criterion(quality):
    diag=np.diagonal(quality)
    diag[diag<1.0]==0
    return diag

def max_std(quality):
    print(np.amin( quality,axis=0))
    print(np.amin( quality,axis=1)  )
    mean_sep=np.mean(quality,axis=1)
    print(mean_sep)
    index=np.argmin(mean_sep)
    print(index)
    print(quality[index])
    print(np.argmax(quality[index]))

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