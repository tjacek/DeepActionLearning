import lasagne
import pickle

class NeuralNetwork(object):
    def __init__(self,hyperparams,out_layer):
        self.hyperparams=hyperparams
        self.out_layer=out_layer

    def get_model(self):
        type_name=type(self).__name__
        data = lasagne.layers.get_all_param_values(self.out_layer)
        return Model(self.hyperparams,data,type_name)
    
    def set_model(self,model):
        lasagne.layers.set_all_param_values(self.out_layer,model.params)

    def __str__(self):
        return str(self.hyperparams)

class Model(object):
    def __init__(self,hyperparams,params,type_name='Convet'):
        self.params=params
        self.hyperparams=hyperparams
        self.type_name=type_name

    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def set_determistic(self):
        if('p' in self.hyperparams):
            self.hyperparams['p']=0.0

    def add_empty_params(self,param_shape):
        print(type(param_shape))
        print(param_shape)
        self.params.append(np.ones(param_shape))