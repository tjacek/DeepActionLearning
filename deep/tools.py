import numpy as np
from sklearn.metrics import classification_report
import seq.io,utils
import basic.extr
import deep.reader,deep.train,deep.convnet

def person_models(in_path,out_path,num_iter=10,n_frames=4):
    X_train,y_train,X_test,y_test=load_data(in_path)
    X,y=X_train,X_train.person
    person_ids=np.unique(y)
    n_persons=person_ids.shape[0]
    frame_preproc=deep.convnet.FramePreproc(n_frames)
    X=frame_preproc(X)
    for person_i in range(n_persons):
        y_i=binarize(y,person_i)
        model_i=make_model(y_i)
        model=deep.train.train_super_model(X,y_i,model_i,num_iter=num_iter)
        out_i=out_path+'/person' + person_ids[person_i]
        model.get_model().save(out_i)

def deep_features(in_path,nn_path,out_path):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    def conv_helper(action_i):
        return conv(action_i.as_array())
    deep_feats=basic.extr.Extractor(conv_helper,feat_fun=False)
    deep_feats(in_path,out_path)	

def train_model(in_path,nn_path=None):
    X_train,y_train,X_test,y_test=load_data(in_path)
    print(X_train.shape)
    model=make_model(y_train)
    model=deep.train.train_super_model(X_train,y_train,model)
    verify_model(y_test,X_test,model)
    if(nn_path):
        model.get_model().save(nn_path)

def load_data(in_path):
    read_actions=seq.io.build_action_reader(img_seq=True,as_dict=False)
    actions=read_actions(in_path)
    train,test=utils.split(actions,lambda action_i: (action_i.person % 2)==1)
    X_train,y_train=as_dataset(train)
    X_test,y_test=as_dataset(test)
    return X_train,y_train,X_test,y_test

def as_dataset(actions):
    X=np.array([np.expand_dims(action_i.as_array(),0) 
                    for action_i in actions])
    y=[action_i.cat-1 for action_i in actions]
    return X,y

def test_model(data_path,nn_path):
    X_train,y_train,X_test,y_test=load_data(data_path)
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    verify_model(y_test,X_test,conv)

def verify_model(y_test,X_test,model):
    y_pred=[model.get_category(x_i) for x_i in X_test]
    print(classification_report(y_test, y_pred,digits=4))

def binarize(y,cat_j):
    return [ 0 if(y_i==cat_j) else 1
               for y_i in y]