import sklearn.svm
import ensemble.data

def make_outliner_detection(handcrafted_path,deep_path,feats):
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    outliners_detections=[train_one_SVM(i,data_i) 
                            for i,data_i in enumerate(datasets)]

def train_one_SVM(i,data_i):
    clf_i = sklearn.svm.OneClassSVM()
    clf_i.fit(data_i.X)
    print("Train one class SVM %d",i)
    return clf_i