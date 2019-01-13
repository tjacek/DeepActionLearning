import numpy as np
from sklearn.metrics import classification_report
import seq.io,utils
import basic.extr
import deep.reader,deep.train,deep.preproc
import deep.convnet,deep.autoconv

def person_models(in_path,out_path,num_iter=10,n_frames=4):
    X,y,frame_preproc=persons_dataset(in_path,n_frames)
    person_ids=np.unique(y)
    print(person_ids)
    n_persons=person_ids.shape[0]
    print(X.shape)
    for i in range(n_persons):
        person_i=person_ids[i]
        y_i=deep.preproc.binarize(y,person_i)
        model_i=deep.convnet.make_model(y_i,"frame")
        model=deep.train.train_super_model(X,y_i,model_i,num_iter=num_iter)
        out_i=out_path+'/person' + str(person_i)
        model.get_model().save(out_i)

def multi_persons_model(in_path,out_path,num_iter=300,n_frames=4):
    X,y,frame_preproc=persons_dataset(in_path,n_frames)
    print(X.shape)
    mp_model=deep.convnet.make_model(y,"frame")
    mp_model=deep.train.train_super_model(X,y,mp_model,num_iter=num_iter)
    mp_model.get_model().save(out_path)

def persons_dataset(in_path,n_frames=4):
    load_data=deep.preproc.LoadData(deep.preproc.person_frames)
    X_train,y_train,X_test,y_test=load_data(in_path)
    X,y=X_train,y_train
    frame_preproc=deep.preproc.FramePreproc(n_frames)
    X=frame_preproc(X)
    y=deep.preproc.cats_to_int(y)
    return X,y,frame_preproc

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

def verify_model(y_test,X_test,model):
    X_test=[np.expand_dims(x_i,0) for x_i in X_test]
    y_pred=[model.get_category(x_i) for x_i in X_test]
    print(classification_report(y_test, y_pred,digits=4))