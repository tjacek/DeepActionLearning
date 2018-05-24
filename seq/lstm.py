import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import utils
import utils.paths.files as files
import numpy as np
import deep.lstm
import utils.data
import deep.reader
import seq
import deep.tools

def make_model(train_dataset,make=True,n_epochs=0,p=0.5):
    if(make):
        hyper_params=deep.lstm.get_hyper_params(train_dataset)
        hyper_params['p']=p
        hyper_params['n_hidden']= 100#get_n_hidden(hyper_params['seq_dim'],hyper_params['n_cats']) 
        model=deep.lstm.compile_lstm(hyper_params)
    else:
        nn_reader=deep.reader.NNReader()
        model= nn_reader(nn_path,p)
    if(n_epochs!=0):
        return train_model(model,train_dataset,epochs=n_epochs)
    return model

def get_n_hidden(seq_dim,n_cats):
    return int((seq_dim+n_cats)/2.0)

def check_model(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    print(utils.data.find_errors(y_pred,test_dataset))
    seq.check_prediction(y_pred,y_true)

class CorrectCond(object):
    def __init__(self, correct,seek_value=True):
        self.correct=correct
        self.seek_value=seek_value

    def __call__(self,dist,i):
        if(self.seek_value):
            return self.correct[i]
        else:
            return (not self.correct[i])

def check_distribution(model,test_dataset):
    x=test_dataset['x']
    y_true=test_dataset['y']
    mask=test_dataset['mask']
    y_pred=[model.get_category(x_i,mask[i])
              for i,x_i in enumerate(x)]
    dists=[model.get_distribution(x_i,mask[i])
              for i,x_i in enumerate(x)]
    correct=[ y_true_i==y_pred_i
              for y_true_i,y_pred_i in zip(y_true,y_pred)]

    dist_correct=[L2(dist_i)
                    for i,dist_i in enumerate(dists)
                      if(correct[i])]
    print(np.average(dist_correct))          
    print(np.average(dist_incorrect))          

def filter_dist(dist,cond):
    return [dist_i
             for i,dist_i in enumerate(dists)
                if(cond(dist,i))]

def L2(dists_i):
    return np.linalg.norm(dists_i,ord=2)

def train_model(model,dataset,epochs=10000):
    print(type(dataset))
    x=get_batches(dataset['x'])
    y=get_batches(dataset['y'])
    print(dataset.keys())
    mask=get_batches(dataset['mask'])
    for j in range(epochs):
        cost=[]
        for i,x_i in enumerate(x):
            y_i=y[i]
            mask_i=mask[i]
            loss_i=model.train(x_i,y_i,mask_i)
            cost.append(loss_i)
        sum_j=sum(cost)/float(len(cost))
        print(str(j) + ' ' + str(sum_j))
    return model

def get_batches(x,batch_size=6):
    n_batches=len(x)/batch_size
    if((len(x) % batch_size)!=0):
        n_batches+=1
    return [x[i*batch_size:(i+1)*batch_size] 
               for i in range(n_batches)]

def get_paths(dir_path,nn,seq='seq',lstm='lstm'):
    path=dir_path + nn+seq 
    nn_path=dir_path + nn+lstm 
    return path,nn_path

def create_dataset(seq_path,nn_path,dataset_format='cp_dataset'):
    test,train=seq.seq_dataset(seq_path,masked=True,dataset_format=dataset_format)
    model=make_model(train,True,n_epochs=200,p=0.5)
    model.get_model().save(str(nn_path))   
    check_model(model,test)

def check_dataset(seq_path,nn_path,n_cats=20):
    preproc=deep.tools.ImgPreproc2D()
    nn_reader=deep.reader.NNReader(preproc)
    for i in range(n_cats):
        seq_i=str(seq_path) + '_' + str(i)
        train,test=seq.seq_dataset(seq_i,masked=True)
        model=nn_reader(nn_path+'_'+str(i))
        check_model(model,test)

def all_lstm(seq_path,out_path,n_cats=20):
    for i in range(n_cats):
        seq_i=str(seq_path) + '_' + str(i)
        out_i=str(out_path) + '_' + str(i)
        create_dataset(seq_i,out_i)

if __name__ == "__main__":
    #path="../../AA_disk3/united_seqs/nn"
    #nn_path="../../AA_disk3/lstms/lstms"
    path="../../Documents/EE/united"
    nn_path="../../Documents/EE/united_lstm"
    #nn_path="../../AA_disk4/lstm"
    create_dataset(path,nn_path,dataset_format='basic_dataset')
    #all_lstm(path,nn_path)
    #check_dataset(path,nn_path)
