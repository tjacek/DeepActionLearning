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

    def recover(self,X):
        if(self.norm):
            X/=self.norm
        return np.array([np.concatenate(x_i,axis=0) 
                            for x_i in X])

def ts_preproc(action_i):
    array_img=action_i.as_array()
    array_img=np.expand_dims(array_img,0)
    return np.expand_dims(array_img,0)

class LoadData(object):
    def __init__(self,as_dataset="persons"):
        img_seq=True
        if(as_dataset=="persons"):
            img_seq=True
            as_dataset=person_frames
        if(as_dataset=="cats"):
            img_seq=True
            as_dataset=cat_frames
        if(as_dataset=="time_series"):
            img_seq=False
            as_dataset=time_series_imgs
        if(as_dataset=="unsuper"):
            img_seq=True
            as_dataset=unsuper_data
        self.read_actions=seq.io.build_action_reader(img_seq=img_seq,as_dict=False)
        self.as_dataset=as_dataset

    def __call__(self,in_path):
        actions=self.read_actions(in_path)
        train,test=standard_split(actions)
        X_train,y_train=self.as_dataset(train)
        X_test,y_test=self.as_dataset(test)
        return X_train,y_train,X_test,y_test

def frame_dataset(in_path,as_dataset="persons",n_frames=4):
    load_data=LoadData(as_dataset)
    X_train,y_train,X_test,y_test=load_data(in_path)
    X,y=X_train,y_train
    frame_preproc=FramePreproc(n_frames)
    X=frame_preproc(X)
    y=cats_to_int(y)
    return X,y,frame_preproc

def person_frames(actions):
    X,y=[],[]
    for action_i in actions:
    	for img_ij in action_i.img_seq:
    	    X.append(img_ij)
    	    y.append(action_i.person)
    return np.array(X),y

def cat_frames(actions):
    X,y=[],[]
    for action_i in actions:
        for img_ij in action_i.img_seq:
            X.append(img_ij)
            y.append(action_i.cat)
    return np.array(X),y

def time_series_imgs(actions):
    X,y=[],[]
    for action_i in actions:
        array_img=action_i.as_array()
        array_img=np.expand_dims(array_img,0)
        print(array_img.shape)
        X.append(array_img)
        y.append(action_i.cat)
    return np.array(X),cats_to_int(y)

def unsuper_data(actions):
    X=[]
    for action_i in actions:
        for img_ij in action_i.img_seq:
            X.append(img_ij)
    return np.array(X),None

def standard_split(actions):
    return utils.split(actions,lambda action_i: (action_i.person % 2)==1)

def cats_to_int(y):
    all_cats=np.unique(y)
    cats_dict={ cat_name:j for j,cat_name in enumerate(all_cats)}
    return [ cats_dict[y_i] for y_i in y]

def binarize(y,cat_j):
    return [ 0 if(y_i==cat_j) else 1
               for y_i in y]