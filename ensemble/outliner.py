import sklearn.svm
import utils,ensemble.data
from joblib import dump, load

class OutlinerWeights(object):
    def __init__(self, outliner_detectors):
        self.outliner_detectors = outliner_detectors

    def __call__(self,x_i):
        return [outliner_i.predict(x_i)[0] 
                    for outliner_i in self.outliner_detectors]
		
def examine(handcrafted_path,deep_path,feats,detectors_path):
    detectors=read_detectors(detectors_path)
    datasets,n_feats=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)    
    def inst_helper(insts):
    	for inst_j in insts:
            sample_j=inst_j.data.reshape(1, -1)
            weights=detectors(sample_j)
            print(inst_j.name +" "+ str(weights))
    def examine_helper(dataset_i):
        insts=datasets[0].to_instances()
        train_insts,test_insts=insts.split()
        print("Train:")
        inst_helper(train_insts.ordered_raw())
        print("###############################")
        print("\n\n\nTest:")
        inst_helper(test_insts.ordered_raw())
    examine_helper(datasets[0])

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