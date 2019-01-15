import utils

class GroupFun(object):
	def __init__(self, fun):
		self.fun = fun
		
    def __init__(nn_path,out_path):
        model_paths=utils.bottom_dirs(nn_path)
        seqs_paths=utils.switch_paths(out_path,model_paths)
        utils.make_dir(out_path)
        for model_path_i,seq_path_i in zip(model_paths,seqs_paths):
            self.fun(model_path_i,seq_path_i)