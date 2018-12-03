import numpy as np
import seq.io,utils
import deep.convnet
from sklearn.metrics import classification_report

def make_dataset(in_path,nn_path=None):
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(in_path)
    train,test=utils.split(actions,lambda action_i: (action_i.person % 2)==1)
    X_train,y_train=as_dataset(train)
    n_cats=np.unique(y_train).shape[0]
    params=deep.convnet.default_params()
    params['n_cats']=n_cats
    model=deep.convnet.compile_convnet(params)
    print(X_train.shape)
    model=train_super_model(X_train,y_train,model)
    X_test,y_test=as_dataset(test)
    y_pred=[model.get_category(x_i) for x_i in X_test]
    print(classification_report(y_test, y_pred,digits=4))
    if(nn_path):
        model.get_model().save(nn_path)

def as_dataset(actions):
    X=np.array([np.expand_dims(action_i.as_array(),0) 
                    for action_i in actions])
    y=[action_i.cat-1 for action_i in actions]
    return X,y

def train_super_model(X,y,model,
                      batch_size=100,num_iter=500):
    print("Num iters " + str(num_iter))
    x_batch,n_batches=get_batch(X,batch_size)
    y_batch,n_batches=get_batch(y,batch_size)
    print(x_batch[0].shape)
    for epoch in range(num_iter):
        cost_e = []
        for i in range(n_batches):
            x_i=x_batch[i]
            y_i=y_batch[i]
            loss_i=model.updates(x_i,y_i)
            cost_e.append(loss_i)
        cost_mean=np.mean(cost_e)
        print(str(epoch) + " "+str(cost_mean))
    return model

def get_batch(imgs,batch_size=10):
    n_batches=get_n_batches(imgs,batch_size)
    batches=[imgs[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    return [np.array(batch_i) for batch_i in batches],n_batches

def get_n_batches(imgs,batch_size=10):
    n_imgs=len(imgs)
    n_batches=n_imgs/batch_size
    if((n_imgs%batch_size)!=0):
        n_batches+=1
    return int(n_batches)