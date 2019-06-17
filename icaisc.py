import ensemble

ens=ensemble.Ensemble()

dataset="mra"
single_exp={'handcrafted_path':"datasets/ICAISC/handcrafted/"+dataset,
           'deep_path':"datasets/ICAISC/deep/selected/" +dataset,
           'feats':(100,0)}

ens(**single_exp)

dataset="mhad"
single_exp={'handcrafted_path':"datasets/ICAISC/handcrafted/"+dataset,
           'deep_path':"datasets/ICAISC/deep/selected/" +dataset,
           'feats':(100,0)}
ens(**single_exp)
