import numpy as np
from sklearn.metrics import classification_report
import seq.io,utils
import basic.extr
import deep.reader,deep.train,deep.convnet,deep.preproc

def person_models(in_path,out_path,num_iter=10,n_frames=4):
    load_data=deep.preproc.LoadData(deep.preproc.person_frames)
    X_train,y_train,X_test,y_test=load_data(in_path)
    X,y=X_train,y_train
    person_ids=np.unique(y)
    print(person_ids)
    n_persons=person_ids.shape[0]
    frame_preproc=deep.preproc.FramePreproc(n_frames)
    X=frame_preproc(X)
    print(X.shape)
    for i in range(n_persons):
        person_i=person_ids[i]
        y_i=binarize(y,person_i)
        model_i=deep.convnet.make_model(y_i,"frame")
        model=deep.train.train_super_model(X,y_i,model_i,num_iter=num_iter)
        out_i=out_path+'/person' + str(person_i)
        model.get_model().save(out_i)

def build_deep_extractor(nn_path,cat_feat=False):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    if(cat_feat):
        def conv_helper(action_i):
            dist_i=conv.get_distribution(action_i.as_array())
            return [dist_i[0]]
    else:
        conv_helper=lambda action_i:conv(action_i.as_array())
    return basic.extr.Extractor(conv_helper,feat_fun=False,img_seq=True)

def train_model(in_path,nn_path=None):
    X_train,y_train,X_test,y_test=load_data(in_path)
    print(X_train.shape)
    model=make_model(y_train)
    model=deep.train.train_super_model(X_train,y_train,model)
    verify_model(y_test,X_test,model)
    if(nn_path):
        model.get_model().save(nn_path)

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