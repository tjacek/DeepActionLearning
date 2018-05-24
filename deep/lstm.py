import lasagne,pickle
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from . import Model
import deep

class LSTM(deep.NeuralNetwork):
    def __init__(self,hyperparams,out_layer,
                     in_var,mask_var,target_var,
                     pred,loss,updates):
        super(LSTM,self).__init__(hyperparams,out_layer)
        self.predict= theano.function([in_var,mask_var],pred)
        self.train = theano.function([in_var, target_var,mask_var],loss, updates=updates)
        self.loss = theano.function([in_var,target_var,mask_var], loss)

    def get_category(self,x,mask):
        return np.argmax(self.get_distribution(x,mask))

    def get_distribution(self,x,mask):
        x=np.expand_dims(x,axis=0)
        mask=np.expand_dims(mask,axis=0)
        return self.predict(x,mask).flatten()

    def dim(self):
        return self.hyperparams['seq_dim']

def compile_lstm(hyper_params,preproc=None):
    l_out,input_vars=make_LSTM(hyper_params)
    prediction = lasagne.layers.get_output(l_out)
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction,input_vars['target_var'])
    loss = loss.mean()
    
    prediction_det = lasagne.layers.get_output(l_out, deterministic=True)

    updates =lasagne.updates.adagrad(loss,params, hyper_params['learning_rate'])
    return LSTM(hyper_params,l_out,
                     input_vars['in_var'],input_vars['mask_var'],input_vars['target_var'],
                     prediction_det,loss,updates)

def make_LSTM(hyper_params):
    print(hyper_params)
    n_batch=None#hyper_params['n_batch']
    max_seq=hyper_params['max_seq']
    seq_dim=hyper_params['seq_dim']
    n_cats=hyper_params['n_cats']
    n_hidden=hyper_params['n_hidden']
    grad_clip = hyper_params['grad_clip']
    l_in = lasagne.layers.InputLayer(shape=(n_batch, max_seq, seq_dim))
    l_mask = lasagne.layers.InputLayer(shape=(n_batch, max_seq))
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)
    l_backward_slice = lasagne.layers.SliceLayer(l_backward, 0, 1)
    l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice])
    l_drop= lasagne.layers.DropoutLayer( lasagne.layers.FlattenLayer(l_sum),
                                         p=hyper_params['p'])
    l_out = lasagne.layers.DenseLayer(
        l_drop, num_units=n_cats, nonlinearity=lasagne.nonlinearities.softmax) 
    input_vars=make_input_vars(l_in,l_mask)
    return l_out,input_vars

def make_input_vars(l_in,l_mask):
    in_var=l_in.input_var
    target_var = T.ivector('targets')
    mask_var=l_mask.input_var
    return {'in_var':in_var,'target_var':target_var,'mask_var':mask_var}

def get_hyper_params(masked_dataset):
    hyper_params=masked_dataset['params']
    hyper_params['n_hidden']=65
    hyper_params['grad_clip']=100
    hyper_params['learning_rate']=0.001
    hyper_params['momentum']=0.9
    hyper_params['p']=0.5
    return hyper_params

def read_lstm(path):
    with open(path, 'r') as f:
        model = pickle.load(f)
    model.hyperparams['p']=0.0
    lstm_model=compile_lstm(model.hyperparams)
    lstm_model.set_model(model)
    return lstm_model   