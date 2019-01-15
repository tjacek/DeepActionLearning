import deep
import pickle
import deep.convnet,deep.autoconv

class NNReader(object):
    def __init__(self,preproc=None):
        self.types = {'Convet':deep.convnet.compile_convnet,
                      'ConvAutorncoder':deep.autoconv.compile_conv_ae}
        
    def __call__(self,in_path, drop_p=0.0,get_hyper=False):
        model=self.__unpickle__(in_path) 
        model.hyperparams['p']=drop_p
        type_reader=self.types[model.type_name]
        neural_net=type_reader(model.hyperparams)
        neural_net.set_model(model)
        if(get_hyper):
            return neural_net,model.hyperparams
        else:
            return neural_net
    
    def __unpickle__(self,in_path):
        with open(str(in_path), 'rb') as f:
            model = pickle.load(f)
        return model