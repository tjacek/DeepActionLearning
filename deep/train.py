import numpy as np
import deep.convnet

def train_super_model(X,y,model,
                      batch_size=100,num_iter=1500):
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