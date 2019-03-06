import sklearn.svm
import utils,ensemble.data
from joblib import dump, load

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
    return detectors