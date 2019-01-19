import numpy as np
import preproc,preproc.agum,preproc.sampling,preproc.normal
import seq.io,seq.tools

def for_ts_network(norm=True,size=128):
    all_funs=[preproc.sampling.SplineUpsampling(size)]
    if(norm):
        all_funs.append(preproc.normal.z_norm)
    pipeline=preproc.Pipeline(all_funs)
    return preproc.Preproc(pipeline)

def basic_agum(in_path,out_path):
    agum_upsampl=preproc.agum.Agumentation([preproc.agum.SamplingAgum()])#,Scale()])
    agum_upsampl(in_path,out_path)

def preproc_concat(paths,out_path,norms,preproc_build=None):
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=True)
    action_sets=[read_actions(path_i) for path_i in paths]
    action_sets=[preproc_build(norm_i)(action_i) 
                    for action_i,norm_i in zip(action_sets,norms)]
    concated_actions=seq.tools.concat_actions(action_sets)
    save_actions=seq.io.ActionWriter(img_seq=False)
    save_actions(concated_actions,out_path)