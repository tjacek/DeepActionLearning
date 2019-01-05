import numpy as np
import preproc,preproc.sampling,preproc.normal

def for_ts_network(size=128):
    upsample=preproc.sampling.SplineUpsampling(size)
    all_funs=[upsample,preproc.normal.z_norm]
    pipeline=preproc.Pipeline(all_funs)
    return preproc.Preproc(pipeline)