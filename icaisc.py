import ensemble

ens=ensemble.Ensemble()
dataset="mra"
main="datasets/ICAISC/"
single_exp={'handcrafted_path':main+"handcrafted/"+dataset,
           'deep_path': main+'/deep/selected/' +dataset,
           'feats':(100,0)}

ens(**single_exp)
