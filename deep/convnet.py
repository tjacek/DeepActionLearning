import numpy as np
import deep,deep.reader
import lasagne
import theano
import theano.tensor as T
from lasagne.regularization import regularize_layer_params, l2, l1

class Convet(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,
                     in_var,target_var,
                     features_pred,pred,loss,updates,preproc=None):
        super(Convet,self).__init__(hyperparams,out_layer)
        self.in_var=in_var
        self.target_var=target_var
        self.__features__=theano.function([in_var],features_pred,allow_input_downcast=True)
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)
        self.preproc=preproc

    def __call__(self,in_img):
        in_img=self.preproc(in_img)  if(self.preproc) else in_img
        return self.__features__(in_img).flatten()
    
    def get_category(self,img):
        dist=self.get_distribution(img)
        return np.argmax(dist)

    def get_distribution(self,x):
        x=self.preproc(x) if(self.preproc) else x
        img_x=self.pred(x)
        return img_x

    def dim(self):
        return self.hyperparams['n_hidden']

def compile_convnet(params):
    if(not params):
        params=frame_network_params(20)
    in_layer,out_layer,hid_layer,all_layers=build_model(params)
    target_var = T.ivector('targets')
    features_pred = lasagne.layers.get_output(hid_layer)
    pred,in_var=get_prediction(in_layer,out_layer)
    if(not 'l1_reg' in params):
        params['l1_reg']=0.001
    l1_reg=params['l1_reg']
    loss=get_loss(pred,in_var,target_var,all_layers,l1_reg)
    updates=get_updates(loss,out_layer)
    return Convet(params,out_layer,
                  in_var,target_var,
                  features_pred,pred,loss,updates)

def build_model(params):
    print(params)
    input_shape=params["input_shape"]
    n1_filters=params["n1_filters"]
    n2_filters=params["n2_filters"]
    filter_size2D=params["filter_size"]
    pool_size2D=params["pool_size"]
    p_drop=params["p"]
    n_cats=params['n_cats']
    n_hidden=params.get('n_hidden',100) 

    in_layer = lasagne.layers.InputLayer(
               shape=input_shape)
               #input_var=input_var)
    conv_layer1 = lasagne.layers.Conv2DLayer(
            in_layer, num_filters=n1_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pool_layer1 = lasagne.layers.MaxPool2DLayer(conv_layer1, pool_size=pool_size2D)
    conv_layer2 = lasagne.layers.Conv2DLayer(
            pool_layer1, num_filters=n2_filters, filter_size=filter_size2D,
            nonlinearity=lasagne.nonlinearities.rectify)
    pool_layer2 = lasagne.layers.MaxPool2DLayer(conv_layer2, pool_size=pool_size2D)
    print(lasagne.layers.count_params(pool_layer2))

    dropout = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool_layer2, p=p_drop),
            num_units= n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.count_params(dropout))
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

def make_model(y,get_params,dim=12):
    n_cats=np.unique(y).shape[0]
    if(get_params=="frame"):
        params=frame_network_params(n_cats)
    elif(get_params=="time_series"):
        params=ts_network_params(n_cats,dim)
    else:    
        params= get_params(n_cats)#deep.convnet.default_params()
    return deep.convnet.compile_convnet(params)

def ts_network_params(n_cats,dim):
    #n_dim,ts_len= dim[0]if(type(dim)==tuple) else dim,128
    return {"input_shape":(None,1,256,6),"n_cats":n_cats,
            "n1_filters":8,"n2_filters":8,"n_hidden":100,
            "filter_size":(8,1),"pool_size":(4,1),"p":0.5, "l1_reg":0.001}

def frame_network_params(n_cats):
    return {"input_shape":(None,4,64,64),"n_cats":n_cats,
            "n1_filters":16,"n2_filters":16,"n_hidden":100,"filter_size":(5,5),
            "pool_size":(4,4),"p":0.5, "l1_reg":0.001,"norm":True}