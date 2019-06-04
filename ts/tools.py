import numpy as np
import ts,ts.agum,ts.sampling,ts.normal
import seq.io,seq.tools,utils
from sets import Set

def for_ts_network(norm=True,size=128):
    all_funs=[ts.sampling.SplineUpsampling(size)]
    if(norm):
        all_funs.append(ts.normal.z_norm)
    pipeline=ts.Pipeline(all_funs)
    return ts.Preproc(pipeline)

def basic_agum(in_path,out_path):
    agum_upsampl=ts.agum.Agumentation([preproc.agum.SamplingAgum()])#,Scale()])
    agum_upsampl(in_path,out_path)

def preproc_concat(in_path,out_path,restr=None,preproc_build=None):
    paths,norms,preproc_build=prepare_concat(in_path,restr,preproc_build)
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=True)
    action_sets=[read_actions(path_i) for path_i in paths]
    action_sets=[preproc_build(norm_i)(action_i) 
                    for action_i,norm_i in zip(action_sets,norms)]
    concated_actions=seq.tools.concat_actions(action_sets)
    save_actions=seq.io.ActionWriter(img_seq=False)
    save_actions(concated_actions,out_path)

def prepare_concat(in_path,restr=None,preproc_build=None):
    paths=utils.top_dirs(in_path)
    if(type(restr)!=list):
        restr=[restr]
    restr=Set(restr)
    norms=[ not (path_i in restr) for path_i in paths]
    if(not preproc_build):
        preproc_build=for_ts_network
    return paths,norms,preproc_build