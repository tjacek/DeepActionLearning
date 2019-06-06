import numpy as np
import ts,ts.agum,ts.sampling,ts.normal
import seq.io,seq.tools,utils
from sets import Set

def for_ts_network(norm=True,size=128):
    #all_funs=[ts.sampling.SplineUpsampling(size)]
    #if(norm):
    #    all_funs.append(ts.normal.z_norm)
    #pipeline=ts.Pipeline(all_funs)
    #return ts.Preproc(pipeline)
    return ts.sampling.SplineUpsampling(size)
    
def basic_agum(in_path,out_path):
    agum_upsampl=ts.agum.Agumentation([preproc.agum.SamplingAgum()])#,Scale()])
    agum_upsampl(in_path,out_path)

def ts_concat(in_path,out_path,preproc=256):
    if(type(preproc)==int):
        preproc=for_ts_network(norm=True,size=preproc)
    dir_paths=utils.bottom_dirs(in_path)
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=True)    
    all_actions=[ read_actions(path_i) for path_i in dir_paths]
    if(preproc):
        all_actions=[{name_j:action_j(preproc,whole_seq=False,feats=True) 
                        for name_j,action_j in action_set_i.items()}
                            for action_set_i in all_actions]
    concated_actions=seq.tools.concat_actions(all_actions).values()
    save_actions=seq.io.ActionWriter(False)
    save_actions(concated_actions,out_path)