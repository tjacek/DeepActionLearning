import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np
import theano
import theano.tensor as T
import lasagne
import tools 
import pickle
from lasagne.regularization import regularize_layer_params, l2, l1
import deep,train
import utils
import utils.imgs as imgs
import utils.text as text
import utils.data as data
import utils.paths,utils.timer,utils.actions.tools
import deep.reader
import gc

#import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu")

class Convet(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,preproc,
                     in_var,target_var,
                     features_pred,pred,loss,updates):
        super(Convet,self).__init__(hyperparams,out_layer)
        self.preproc=preproc
        self.in_var=in_var
        self.target_var=target_var
        self.__features__=theano.function([in_var],features_pred)
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def __call__(self,in_img):
        print(type(in_img))
        img4D=self.preproc.apply(in_img)
        return self.__features__(img4D).flatten()
    
    def get_category(self,img):
        dist=self.get_distribution(img)
        return np.argmax(dist)

    def get_distribution(self,x):
        if(len(x.shape)!=4):
            img4D=self.preproc.apply(x)
        else:
            img4D=x
        img_x=self.pred(img4D).flatten()
        return img_x

    def dim(self):
        return self.hyperparams['n_hidden']

def compile_convnet(params,preproc):
    in_layer,out_layer,hid_layer,all_layers=build_model(params)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    if(not 'l1_reg' in params):
        params['l1_reg']=0.001
    l1_reg=params['l1_reg']
    loss=get_loss(pred,in_var,target_var,all_layers,l1_reg)
    updates=get_updates(loss,out_layer)
    return Convet(params,out_layer,preproc,
                  in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params):
    print(params)
    input_shape=params["input_shape"]
    n_filters=params["num_filters"]
    filter_size2D=params["filter_size"]
    pool_size2D=params["pool_size"]
    p_drop=params["p"]
    n_cats=params['n_cats']
    n_hidden=params.get('n_hidden',300) 

    in_layer = lasagne.layers.InputLayer(
               shape=input_shape)
               #input_var=input_var)
    conv_layer1 = lasagne.layers.Conv2DLayer(
            in_layer, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pool_layer1 = lasagne.layers.MaxPool2DLayer(conv_layer1, pool_size=pool_size2D)
    conv_layer2 = lasagne.layers.Conv2DLayer(
            pool_layer1, num_filters=n_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify)
    pool_layer2 = lasagne.layers.MaxPool2DLayer(conv_layer2, pool_size=pool_size2D)
    dropout = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool_layer2, p=p_drop),
            num_units= n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    out_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(dropout, p=p_drop),
            num_units=int(n_cats),
            nonlinearity=lasagne.nonlinearities.softmax)
    all_layers={"in":in_layer, "conv1":conv_layer1,"pool":pool_layer1,
                "conv2":conv_layer2,"pool2":pool_layer2,
                "hidden":dropout,"out":out_layer }
    return in_layer,out_layer,dropout,all_layers

def get_prediction(in_layer,out_layer):
    in_var=in_layer.input_var
    prediction = lasagne.layers.get_output(out_layer)
    return prediction,in_var

def get_loss(prediction,in_var,target_var,all_layers,l1_reg=0.001):    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    l_hid=all_layers["out"]
    reg_param=0.001
    if(l1_reg>0.0):
        l1_penalty = regularize_layer_params(l_hid, l1) * reg_param
        return loss + l1_penalty
    else:
        return loss

def get_updates(loss,out_layer):
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    return updates

def default_params():
    return {"input_shape":(None,2,60,60),"num_filters":16,"n_hidden":100,
              "filter_size":(5,5),"pool_size":(4,4),"p":0.5, "l1_reg":0.001}

def get_model(n_cats,preproc,nn_path=None, params=None, compile=True,l1_reg=True,model_p=0.1):
    if(nn_path==None):
        compile=True
    if(compile):
        if(params==None):
            params=default_params()
        old_shape=params['input_shape']
        params['input_shape']=(old_shape[0],preproc.dim,old_shape[2],old_shape[3])
        params['n_cats']= n_cats#data.get_n_cats(y)
        return compile_convnet(params,preproc)
    else:  
        nn_reader=deep.reader.NNReader(preproc)
        return nn_reader(nn_path,model_p)

def binarize(cat,y):
    print(y)
    return [ int(cat==y_i)
                for y_i in y]

#@utils.timer.clock
def experiment(x,y,preproc,nn_path,n_models,n_iters):
    if(type(n_models)==int):
        n_models=range(n_models)
    for i in n_models:
        nn_path_i=nn_path+'_'+str(i)
        b_y=binarize(i,y)
        n_cats=data.get_n_cats(b_y)
        model=get_model(n_cats,preproc,nn_path_i,compile=True,model_p=0.5)
        train.test_super_model(x,b_y,model,num_iter=n_iters)
        model.get_model().save(nn_path_i)
        gc.collect()

def single_exp(in_path,nn_path):
    preproc=tools.ImgPreprocProj()
    imgset=imgs.make_imgs(img_path,norm=True)
    extract_cat=data.ExtractCat()
    x,y=imgs.to_dataset(imgset,extract_cat,preproc)
    model=get_model(10,preproc,nn_path,compile=False,model_p=0.5)
    train.test_super_model(x,y,model,num_iter=100)
    model.get_model().save(nn_path)

def conv_features(img_path,nn_path,out_path):
    preproc=tools.ImgPreprocProj()
    imgset=imgs.make_imgs(img_path,norm=True)
    extract_cat=data.ExtractCat()
    x,y=imgs.to_dataset(imgset,extract_cat,preproc)  
    model=get_model(10,preproc,nn_path,compile=False,model_p=0.0)

    transform_actions=utils.action.tools.ActionTransform()
    transform_actions(img_path,out_path,model)

if __name__ == "__main__":
    img_path="../../Documents/EE/full"
    out_path="../../Documents/EE/seq"

    nn_path="../../Documents/EE/nn"
    single_exp(img_path,nn_path)

    #conv_features(img_path,nn_path,out_path)
    #print("read")

    #print(len(imgset))
    #extract_cat=data.ExtractCat()
    #x,y=imgs.to_dataset(imgset,extract_cat,preproc)
    #experiment(x,y,preproc,nn_path,20,500)