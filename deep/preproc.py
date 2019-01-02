import numpy as np
import utils,seq.io

class FramePreproc(object):
    def __init__(self,dim,norm=255):
        self.dim=dim
        self.norm= (1.0/float(norm)) if(type(norm)==int) else None

    def __call__(self,X):
        X=[np.vsplit(x_i,self.dim) for x_i in X]
        X=[np.stack(x_i,0) for x_i in X]
        X=np.array(X)
        if(self.norm):
            X=X.astype(float)
            X*=self.norm
        return X

class LoadData(object):
    def __init__(self,as_dataset="person"):
        if(as_dataset=="person"):
            as_dataset=person_frames
        if(as_dataset=="time_series"):
            as_dataset=time_series_imgs
        self.read_actions=seq.io.build_action_reader(img_seq=True,as_dict=False)
        self.as_dataset=as_dataset

    def __call__(self,in_path):
        actions=self.read_actions(in_path)
        train,test=utils.split(actions,lambda action_i: (action_i.person % 2)==1)
        X_train,y_train=self.as_dataset(train)
        X_test,y_test=self.as_dataset(test)
        return X_train,y_train,X_test,y_test

def person_frames(actions):
    X,y=[],[]
    for action_i in actions:
    	for img_ij in action_i.img_seq:
    	    X.append(img_ij)
    	    y.append(action_i.person)
    return np.array(X),y

def time_series_imgs(actions):
    X,y=[],[]
    for action_i in actions:
        X.append(action_i.as_array())
        y.append(action_i.cat)
    return np.array(X),y
