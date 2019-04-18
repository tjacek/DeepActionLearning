import utils

class GroupFun(object):
    def __init__(self, fun,dirs=False):
        self.fun = fun
        self.dirs=dirs
		
    def __call__(self,nn_path,out_path,other_args=None):
        model_paths=utils.bottom_dirs(nn_path) if(self.dirs) else utils.bottom_files(nn_path)
        seqs_paths=utils.switch_paths(out_path,model_paths)
        utils.make_dir(out_path)
        for model_path_i,seq_path_i in zip(model_paths,seqs_paths):
            if(other_args):
                other_args["in_path"]=model_path_i
                other_args["out_path"]=seq_path_i
                self.fun(**other_args)
            else:
                self.fun(model_path_i,seq_path_i)