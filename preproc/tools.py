import numpy as np
import preproc,preproc.sampling,preproc.normal

def for_ts_network(size=128,norm=True):
    all_funs=[preproc.sampling.SplineUpsampling(size)]
    if(norm):
        all_funs.append(preproc.normal.z_norm)
    pipeline=preproc.Pipeline(all_funs)
    return preproc.Preproc(pipeline)