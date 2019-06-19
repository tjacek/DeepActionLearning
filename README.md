Code and data for paper "Ensemble of Classifiers Using CNN and 
Hand-Crafted Features for Depth-Based Action Recognition"

Data description:

     datasets/ICAISC/deep - features from CNN.

     datasets/ICAISC/deep/raw - all deep features

     datasets/ICAISC/deep/selected - deep features selected by RFE algorithm.

     datasets/ICAISC/handcrafted/ - handcrafted features

     All directories contains data for two datasets:

          UT-MHAD - www.utdallas.edu/~kehtar/UTD-MHAD.html

          MSR Action3D Dataset - https://documents.uow.edu.au/~wanqing/#Datasets

Result reproduction:

python icaisc.py

Code for result reproduction:
```
import ensemble
dataset="mhad"
single_exp={'handcrafted_path':"datasets/ICAISC/handcrafted/"+dataset,
           'deep_path':"datasets/ICAISC/deep/selected/" +dataset,
           'feats':(100,0)}
ens=ensemble.Ensemble()

ens(**single_exp)
```

handcrafted_path- path to handcrafted features from CNN
deep_path- path to features from CNN
feats- first element of tuple is number of selected handcrafted features and secund number of selected deep features
if any number is zero there is no selection.