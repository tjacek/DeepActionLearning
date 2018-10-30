import numpy as np
from sklearn import preprocessing
import basic.instances

class Dataset(object):
    def __init__(self,X,y,persons,names):
        self.X=X
        self.y=y
        self.persons=persons
        self.names=names

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]	
    
    def norm(self):
        self.X=preprocessing.scale(self.X)

    def split(self):
        insts=self.to_instances()
        train,test=insts.split()
        return to_dataset(train),to_dataset(test)

    def to_instances(self):
        n_insts=len(self)
        insts=[]
        for i in range(n_insts):
            x_i,y_i,person_i,name_i=self.X[i],self.y[i],self.persons[i],self.names[i]
            inst_i=dataset.instances.Instance(x_i,y_i,person_i,name_i)
            insts.append(inst_i)
        return instances.InstsGroup(insts)

def read_dataset(in_path):
    if(type(in_path)==list):
        datasets=[read_dataset(path_i) for path_i in in_path]
        return unify_datasets(datasets)
    insts=basic.instances.from_files(in_path)
    return to_dataset(insts)

def to_dataset(insts):
    X=np.array(insts.data())
    y,persons,names=insts.cats(),insts.persons(),insts.names()
    return Dataset(X=X,y=y,persons=persons,names=names)

def unify_datasets(datasets):
    datasets=[dataset_i 
                for dataset_i in datasets
                    if(not dataset_i is None)]
    if(len(datasets)==1):
        return datasets[0]
    feats=[ dataset_i.X for dataset_i in datasets]
    united_X=np.concatenate(feats,axis=1)
    first=datasets[0]
    return Dataset(united_X,first.y,first.persons,first.names)