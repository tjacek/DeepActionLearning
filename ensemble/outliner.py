import numpy as np
import sklearn.svm
import utils,ensemble.data
from joblib import dump, load

class OutlinerWeights(object):
    def __init__(self, outliner_detectors):
        self.outliner_detectors = outliner_detectors

    def __call__(self,all_samples_i):
        weights=np.array([outliner_j.predict(sample_ij)[0] 
                    for sample_ij, outliner_j in zip(all_samples_i,self.outliner_detectors)])
        weights[weights==(-1)]=0
        return weights

    def get_weights(self,ensemble_dataset):
        return { name_i.strip(): self(samples_i) 
                    for name_i,samples_i in ensemble_dataset.items()}

def get_weights(handcrafted_path,deep_path,feats,outliner_path):
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    ensemble_dataset=ensemble.data.to_ensemble_samples(datasets)
    outliner_weights=read_detectors(outliner_path)
    weights=outliner_weights.get_weights(ensemble_dataset['test'])
    zero_weights(weights)
    return weights

def make_outliner_detectors(handcrafted_path,deep_path,feats,out_path):
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    outliner_detectors=[train_one_SVM(i,data_i) 
                            for i,data_i in enumerate(datasets)]
    utils.make_dir(out_path)
    for i,detector_i in enumerate(outliner_detectors):
        path_i= '%s/feature_set_%d.joblib' % (out_path,i)
        dump(detector_i, path_i) 

def train_one_SVM(i,data_i):
    clf_i = sklearn.svm.OneClassSVM()
    train_i,test_i=data_i.split()
    clf_i.fit(train_i.X)
    print("Train one class SVM %d" % i)
    return clf_i

def read_detectors(in_path):
    all_paths=utils.bottom_files(in_path)
    print(all_paths)
    detectors=[ load(path_i) for path_i in all_paths]
    return OutlinerWeights(detectors)

def examine(handcrafted_path,deep_path,feats,detectors_path):
    detectors=read_detectors(detectors_path)
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)    
    ensemble_dataset=ensemble.data.to_ensemble_samples(datasets)
    def examine_helper(dict_i):
        weights_sum=0
        for name_j,samples_j in dict_i.items():
            weights_j=detectors(samples_j)
            weights_sum+=sum(weights_j)
    	    print(name_j + str(sum(weights_j)))
        return weights_sum   
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\nTrain ")
    print(examine_helper(ensemble_dataset['train']))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\nTest ")
    print(examine_helper(ensemble_dataset['test']))       

def zero_weights(weights):
    n_dims=weights.values()[0].shape[0]
    zero_names=[name_i 
                    for name_i,weights_i in weights.items()
                        if(np.mean(weights_i)==0.0)]
    print(len(zero_names))
    for name_i in zero_names:
        weights[name_i]= np.ones((n_dims),dtype=float)
    return weights    	