import numpy as np
from sklearn.metrics import classification_report
import seq.io,utils
import basic.extr
import deep.reader,deep.train,deep.preproc
import deep.convnet,deep.autoconv,os

class BinaryModels(object):
    def __init__(self,as_dataset="cats",name="nn",n_frames=4):
        self.n_frames=n_frames
        self.name=name
        self.as_dataset=as_dataset

    def __call__(self,in_path,out_path,num_iter=10):
        X,y,frame_preproc=deep.preproc.frame_dataset(in_path,
                    as_dataset=self.as_dataset,n_frames=self.n_frames)
        person_ids=np.unique(y)
        model_paths=[out_path+'/'+self.name+str(i) for i in person_ids]
        binary_datasets=[deep.preproc.binarize(y,person_i) 
                            for person_i in person_ids]
        for y_i,out_i in zip(binary_datasets,model_paths):
            train_model(X,y_i,out_i,num_iter)                   

def multi_persons_model(in_path,out_path,num_iter=970,n_frames=4):
    X,y,frame_preproc=deep.preproc.frame_dataset(in_path,as_dataset="persons",n_frames=n_frames)
    train_model(X,y,out_path,num_iter)

def train_model(X,y_i,out_i,num_iter):
    if(os.path.isfile(out_i)):
        model_i=deep.reader.NNReader()(out_i)
    else:
        model_i=deep.convnet.make_model(y_i,"frame")
    model=deep.train.train_super_model(X,y_i,model_i,num_iter=num_iter)
    model.get_model().save(out_i)

def train_ts_network(in_path,nn_path,num_iter=1500):
    load_data=deep.preproc.LoadData("time_series")
    X_train,y_train,X_test,y_test=load_data(in_path)
    print(X_train.shape)
    ts_network=deep.convnet.make_model(y_train,"time_series",dim=16)
    ts_network=deep.train.train_super_model(X_train,y_train,ts_network,num_iter=num_iter)
    verify_model(y_test,X_test,ts_network)
    ts_network.get_model().save(nn_path)

def train_autoconv(in_path,nn_path,num_iter=1500,n_frames=4):
    load_data=deep.preproc.LoadData("unsuper")
    X_train,y_train,X_test,y_test=load_data(in_path)
    frame_preproc=deep.preproc.FramePreproc(n_frames)
    X_train=frame_preproc(X_train)
    print(X_train.shape)
    autoencoder=deep.autoconv.make_autoconv()
    autoencoder=deep.train.train_unsuper_model(X_train,autoencoder,num_iter=num_iter)
    autoencoder.get_model().save(nn_path)

def reconstruct_autoconv(in_path,nn_path,out_path,n_frames=4):
    reader=deep.reader.NNReader()
    autoencoder=reader(nn_path)
    frame_preproc=deep.preproc.FramePreproc(n_frames)
    def autoconv_helper(img_seq):
        img_seq=frame_preproc(img_seq)
        rec_seq=autoencoder.reconstructed(img_seq)
        return autoencoder.recover(rec_seq)
    seq.io.transform_actions(in_path,out_path,transform,
                      img_in=True,img_out=True,whole_seq=True)

def verify_model(y_test,X_test,model):
    X_test=[np.expand_dims(x_i,0) for x_i in X_test]
    y_pred=[model.get_category(x_i) for x_i in X_test]
    print(classification_report(y_test, y_pred,digits=4))