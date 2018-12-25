import numpy as np 
import seq.io

def show_scale(in_path):
    actions=get_actions(in_path)
    train,test=actions.select(as_group=True)
    train,test=train.as_array(),test.as_array()
    print(np.amax(train,axis=0))
    print(np.amax(test,axis=0))
    print(np.amin(train,axis=0))
    print(np.amin(test,axis=0))

def show_norm(in_path):
    actions=get_actions(in_path)
    feats=actions.as_array()
    print(np.mean(feats,axis=0))
    print(np.std(feats,axis=0))

def get_actions(in_path):
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False,as_group=True)
    return read_actions(in_path)

if __name__ == "__main__":
    show_norm("../wrap/mra_preproc/all")#mra/basic/data")
