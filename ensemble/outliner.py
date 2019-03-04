import sklearn.svm
import ensemble.data

def make_outliner_detectors(handcrafted_path,deep_path,feats):
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    outliners_detections=[train_one_SVM(i,data_i) 
                            for i,data_i in enumerate(datasets)]

def train_one_SVM(i,data_i):
    clf_i = sklearn.svm.OneClassSVM()
    train_i,test_i=data_i.split()
    clf_i.fit(train_i.X)
    print("Train one class SVM %d" % i)
    return clf_i