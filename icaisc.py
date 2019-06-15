import ensemble.exper

dataset="mhad"
clf_type="LR"
ens,ens_name=ensemble.exper.build_ensemble(dataset=dataset,clf_type=clf_type)

main="datasets/ICAISC/"
single_exp={'handcrafted_path':main+"handcrafted/"+dataset,
           'deep_path': main+'/deep/selected/' +dataset,
           'feats':(100,0)}

ens(**single_exp)
