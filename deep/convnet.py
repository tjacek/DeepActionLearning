import numpy as np
import deep,deep.reader
import lasagne
import theano
import theano.tensor as T
from lasagne.regularization import regularize_layer_params, l2, l1

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
        img3D=self.preproc(in_img)
        img4D=np.expand_dims(img3D,0)
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
    n_hidden=params.get('n_hidden',100) 

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
    return {"input_shape":(None,2,64,64),"num_filters":16,"n_hidden":100,
              "filter_size":(5,5),"pool_size":(4,4),"p":0.5, "l1_reg":0.001}

def get_model(n_cats,preproc,nn_path=None, params=None,l1_reg=True,model_p=0.5):
    if(nn_path is None):
        if(params==None):
            params=default_params()
        old_shape=params['input_shape']
        params['input_shape']=(old_shape[0],preproc.dim,old_shape[2],old_shape[3])
        params['n_cats']= n_cats#data.get_n_cats(y)
        return compile_convnet(params,preproc)
    else:  
        nn_reader=deep.reader.NNReader(preproc)
        return nn_reader(nn_path,model_p)
        